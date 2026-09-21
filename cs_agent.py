"""
PREPARE CS Agent — AI Customer Service dengan RAG + Tool Calling (Gemini)

Alur:
  Customer bertanya
      -> LLM (Gemini) membaca pertanyaan + daftar tools
      -> LLM memutuskan: jawab langsung, ATAU panggil tool
           - cari_produk        (RAG: cari info di katalog pakai embedding)
           - cek_stok           (data inventory)
           - cek_status_order   (data order / "API marketplace")
           - eskalasi_ke_cs_manusia (buat tiket untuk tim CS)
      -> Kode kita menjalankan tool, hasilnya dikirim balik ke LLM
      -> Loop sampai LLM memberi jawaban final ke customer

Jalankan:  venv\\Scripts\\python cs_agent.py
"""

import json
import math
import os
import re
from datetime import datetime

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()
# Matikan retry bawaan SDK (yang bisa menunggu lama saat rate limit) —
# kita tangani sendiri dengan fallback ke model lain di CSAgent.call_llm().
client = genai.Client(
    http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
)

# Urutan model: kalau model pertama kena rate limit (429), otomatis pindah ke berikutnya.
# Free tier Gemini hanya ~5 request/menit per model, jadi fallback ini penting untuk demo.
# Model "flash-lite" dipilih duluan: ~1 detik per respons, cukup pintar untuk tool calling CS.
CHAT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-3.6-flash"]
EMBED_MODEL = "gemini-embedding-001"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCT_FILE = os.path.join(BASE_DIR, "product_data.txt")
ORDERS_FILE = os.path.join(BASE_DIR, "data", "orders.json")
TICKETS_FILE = os.path.join(BASE_DIR, "data", "tickets.log")
EMBED_CACHE = os.path.join(BASE_DIR, "data", "embeddings_cache.json")


# =====================================================================
# 1. RAG — Retrieval-Augmented Generation
#    Dokumen dipecah jadi potongan (chunk) -> diubah jadi vektor (embedding)
#    -> saat ada pertanyaan, cari chunk yang maknanya paling mirip.
#    Di production, bagian penyimpanan vektor ini diganti vector database
#    (Chroma / pgvector / Pinecone). Di sini disimpan di memori + file cache.
# =====================================================================

def load_chunks():
    """Pecah katalog: 1 chunk per produk, plus info perusahaan & kebijakan toko."""
    with open(PRODUCT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r"\n(?==== PRODUK|\[INFORMASI PERUSAHAAN\]|\[KEBIJAKAN TOKO\])", text)
    return [p.strip() for p in parts if p.strip() and not p.strip().startswith("[KATALOG")]


def embed(texts, task_type):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]


def build_index():
    """Buat embedding untuk semua chunk. Di-cache supaya tidak memanggil API tiap start."""
    chunks = load_chunks()
    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("chunks") == chunks:
            return list(zip(chunks, cache["vectors"]))
    vectors = embed(chunks, "RETRIEVAL_DOCUMENT")
    with open(EMBED_CACHE, "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "vectors": vectors}, f)
    return list(zip(chunks, vectors))


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


INDEX = build_index()


# =====================================================================
# 2. TOOLS — fungsi Python yang boleh dipanggil oleh AI
#    Docstring & type hint dibaca Gemini untuk memahami kapan tool dipakai.
# =====================================================================

def cari_produk(query: str) -> dict:
    """Cari informasi produk PREPARE, kebijakan toko (retur, ongkir, pembayaran, jam CS),
    atau info perusahaan berdasarkan kebutuhan customer. Contoh query: 'kulit berminyak',
    'rambut rontok', 'kebijakan retur'."""
    q_vec = embed([query], "RETRIEVAL_QUERY")[0]
    scored = sorted(INDEX, key=lambda item: cosine_similarity(q_vec, item[1]), reverse=True)
    return {"hasil": [chunk for chunk, _ in scored[:3]]}


def cek_stok(nama_produk: str) -> dict:
    """Cek stok terkini sebuah produk PREPARE. Contoh nama_produk: 'pomade', 'hair tonic'."""
    for chunk in load_chunks():
        nama = re.search(r"Nama: (.+)", chunk)
        stok = re.search(r"Stok: (.+)", chunk)
        if nama and stok and nama_produk.lower() in chunk.lower():
            return {"produk": nama.group(1), "stok": stok.group(1)}
    return {"error": f"Produk '{nama_produk}' tidak ditemukan di katalog."}


def cek_status_order(order_id: str) -> dict:
    """Cek status pesanan / pengiriman customer berdasarkan nomor order (format: INV001)."""
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        orders = json.load(f)
    order = orders.get(order_id.strip().upper())
    if not order:
        return {"error": f"Order {order_id} tidak ditemukan. Minta customer cek ulang nomor order."}
    return {"order_id": order_id.upper(), **order}


def eskalasi_ke_cs_manusia(alasan: str, ringkasan_masalah: str) -> dict:
    """Teruskan ke tim CS manusia. WAJIB dipakai untuk: permintaan refund/retur,
    keluhan reaksi kulit/alergi, komplain serius, atau customer yang marah."""
    ticket_id = "TKT-" + datetime.now().strftime("%Y%m%d%H%M%S")
    with open(TICKETS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {ticket_id} | {alasan} | {ringkasan_masalah}\n")
    return {"ticket_id": ticket_id, "status": "Tiket dibuat, tim CS akan menghubungi dalam 1x24 jam kerja."}


TOOLS = {f.__name__: f for f in [cari_produk, cek_stok, cek_status_order, eskalasi_ke_cs_manusia]}


# =====================================================================
# 3. SYSTEM PROMPT + GUARDRAILS
# =====================================================================

SYSTEM_PROMPT = """Kamu adalah "Rara", asisten customer service virtual untuk PREPARE,
brand perawatan diri pria. Gaya bicara: ramah, santai tapi sopan, bahasa Indonesia, singkat.

Aturan:
1. Info produk, harga, bahan, kebijakan toko -> SELALU pakai tool cari_produk dulu. Jangan menebak.
2. Pertanyaan stok -> pakai cek_stok. Pertanyaan pesanan -> pakai cek_status_order.
   Jika customer menanyakan pesanan tapi belum menyebut nomor order, JANGAN eskalasi dan
   jangan panggil tool — tanyakan dulu nomor ordernya (format INV001).
3. Jawab HANYA berdasarkan hasil tool. JANGAN menambah detail yang tidak tertulis di hasil tool
   (ukuran/ml, promo, diskon, waktu restock, dll). Kalau info tidak ada, katakan jujur tidak tahu.
4. JANGAN membuat klaim medis (menyembuhkan, dijamin, pasti berhasil). Untuk pertanyaan soal
   hasil/efektivitas, jelaskan fungsi produk sesuai katalog, katakan hasil tiap orang bisa berbeda,
   dan untuk kondisi medis (kebotakan, jerawat parah, alergi) sarankan konsultasi ke dokter.
5. Refund/retur, reaksi kulit/alergi, atau komplain serius -> WAJIB pakai eskalasi_ke_cs_manusia,
   lalu sampaikan nomor tiketnya. Untuk reaksi kulit, sarankan hentikan pemakaian.
6. Kalau produk habis, tawarkan alternatif produk yang relevan jika ada.
7. Tolak dengan sopan pertanyaan di luar topik PREPARE."""


# =====================================================================
# 4. AGENT LOOP
# =====================================================================

class CSAgent:
    def __init__(self, on_tool_call=None):
        self.history = []  # riwayat percakapan (memory jangka pendek)
        self.on_tool_call = on_tool_call or (lambda name, args, result: None)
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=list(TOOLS.values()),
            thinking_config=types.ThinkingConfig(thinking_level="low"),  # respons lebih cepat
            # Kita jalankan tool secara manual supaya alur agent terlihat jelas
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def call_llm(self):
        """Panggil LLM dengan fallback model jika kena rate limit / server sibuk."""
        last_error = None
        for model in CHAT_MODELS:
            try:
                return client.models.generate_content(
                    model=model, contents=self.history, config=self.config
                )
            except errors.APIError as e:
                if e.code not in (429, 500, 503):
                    raise
                last_error = e
        raise last_error

    def chat(self, user_message: str, max_steps: int = 5) -> str:
        self.history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

        for _ in range(max_steps):
            response = self.call_llm()
            self.history.append(response.candidates[0].content)

            # Tidak ada tool yang diminta -> ini jawaban final
            if not response.function_calls:
                return response.text

            # LLM minta tool -> jalankan, kirim hasilnya balik
            tool_parts = []
            for call in response.function_calls:
                args = dict(call.args or {})
                try:
                    result = TOOLS[call.name](**args)
                except Exception as e:
                    result = {"error": str(e)}
                self.on_tool_call(call.name, args, result)
                tool_parts.append(types.Part.from_function_response(name=call.name, response=result))
            self.history.append(types.Content(role="user", parts=tool_parts))

        return "Maaf, saya butuh bantuan tim CS untuk ini. Mohon tunggu sebentar ya."


def print_tool_call(name, args, result):
    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"   🔧 memanggil {name}({arg_str})")


def main():
    print("=" * 55)
    print("🧴 PREPARE CS Agent  —  RAG + Tool Calling (Gemini)")
    print("=" * 55)
    print("Ketik 'keluar' untuk berhenti.\n")
    agent = CSAgent(on_tool_call=print_tool_call)
    while True:
        pesan = input("Customer: ").strip()
        if not pesan:
            continue
        if pesan.lower() == "keluar":
            print("Rara: Terima kasih sudah menghubungi PREPARE! 👋")
            break
        try:
            print(f"Rara: {agent.chat(pesan)}\n")
        except Exception as e:
            print(f"[!] Error: {e}\n")


if __name__ == "__main__":
    main()
