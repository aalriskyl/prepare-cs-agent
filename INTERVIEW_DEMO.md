# 🎯 Naskah Demo Interview — PREPARE CS Agent

## Cara menjalankan (cek 10 menit sebelum masuk ruangan)
```
venv\Scripts\streamlit run app.py        ← UI chat (utama, lebih menarik)
venv\Scripts\python cs_agent.py          ← versi terminal (cadangan)
```
Butuh internet. **Siapkan hotspot HP** + **rekaman layar demo** (Win+Alt+R) sebagai cadangan kalau internet/API bermasalah.

---

## 1. Pitch 30 detik
> "Saya lihat PREPARE ingin memakai AI untuk operasional. Jadi semalam saya bikin prototipe
> **AI customer service agent** pakai data katalog PREPARE. Agent-nya bisa menjawab pertanyaan
> produk dari katalog (**RAG**), cek stok dan status pesanan lewat **tool calling**, dan otomatis
> **eskalasi ke CS manusia** untuk kasus sensitif seperti refund atau reaksi kulit.
> Tujuannya: mengurangi beban CS untuk pertanyaan berulang, tanpa AI mengarang jawaban."

## 2. Arsitektur (gambar di kertas/whiteboard kalau bisa)
```
 Customer (nanti: WhatsApp / Shopee / IG)
        │
        ▼
 ┌──────────────────────────┐
 │  LLM Agent (Gemini)       │  ← system prompt + guardrails
 │  memutuskan: jawab / tool │
 └────────────┬─────────────┘
              │ tool calling
   ┌──────────┼──────────────┬──────────────────┐
   ▼          ▼              ▼                  ▼
cari_produk  cek_stok   cek_status_order   eskalasi_ke_cs_manusia
 (RAG:        (inventory)  (orders / API      (buat tiket untuk
 embedding +                marketplace)       tim CS)
 similarity)
              │
              ▼
   hasil tool → balik ke LLM → jawaban final ke customer
```

## 3. Urutan demo (± 3 menit) — sambil tunjuk sidebar "Log Tool Calls"
| # | Ketik ini | Yang ditunjukkan |
|---|---|---|
| 1 | Kulit aku berminyak dan sering jerawatan, ada produk yang cocok? | **RAG**: cari produk berdasarkan *makna*, bukan kata kunci |
| 2 | Pomade stoknya masih ada? | **Tool** cek stok → stok habis, dijawab jujur |
| 3 | Pesanan aku kok belum sampai ya | Agent **bertanya nomor order** dulu, tidak asal jawab |
| 4 | INV002 | **Tool** cek order + agent **ingat konteks** percakapan |
| 5 | Muka aku merah setelah pakai face wash, mau refund | **Guardrail**: sarankan stop pemakaian + **eskalasi**, dapat nomor tiket |
| 6 | Bisa bikinin resep rendang? | Menolak topik di luar PREPARE |

Kalimat kunci saat demo:
- "Lihat di sidebar — ini **keputusan AI-nya**: dia memilih sendiri tool mana yang dipanggil."
- "Semua jawaban diambil dari data, bukan dikarang."

## 4. Cerita "masalah yang saya temukan & perbaiki" (ini yang bikin kamu terlihat engineer)
1. **Halusinasi** — saat testing, AI menyebut "ukuran 100ml" padahal tidak ada di katalog.
   → Saya perketat system prompt: dilarang menambah detail di luar hasil tool.
2. **Rate limit** — free tier Gemini cuma ~5 request/menit per model, satu pertanyaan bisa 2–3 request.
   → Saya buat **fallback otomatis ke model lain** kalau kena error 429/503.
3. **Latency** — model besar butuh 20–100 detik per jawaban. Saya benchmark, pindah ke model
   *flash-lite* dengan thinking rendah → **1–3 detik**, kualitas tool calling tetap benar.
4. **Eskalasi berlebihan** — "pesanan belum sampai" tanpa nomor order langsung dieskalasi.
   → Diperbaiki: agent tanya nomor order dulu.

## 5. "Kalau dibawa ke production, apa yang perlu ditambah?"
- **Channel**: webhook WhatsApp Business API / Shopee / Tokopedia chat → backend FastAPI.
- **Data asli**: tool terhubung ke database order & inventory / API marketplace (bukan file JSON).
- **Vector database** (pgvector / Chroma) kalau dokumen banyak (FAQ, SOP, ribuan produk).
- **Evaluasi**: kumpulan 50–100 pertanyaan uji, cek akurasi tiap kali prompt/model diubah.
- **Monitoring**: log semua percakapan & tool call, ukur % yang dieskalasi, kepuasan customer.
- **Biaya**: model kecil untuk pertanyaan sederhana, caching, batas token.
- **Keamanan**: verifikasi identitas sebelum menampilkan data order (nama/no HP), jangan bocorkan data customer lain.

## 6. Use case lain yang bisa kamu tawarkan
- **Generator konten marketing**: caption IG/TikTok & deskripsi marketplace sesuai brand voice.
- **Analisis review**: klasifikasi ribuan review marketplace (sentimen + topik keluhan) → laporan mingguan.
- **Asisten data internal**: manajer tanya "penjualan minggu lalu?" → AI buat query SQL → jawab.

## 7. Pertanyaan yang mungkin muncul
- **RAG vs fine-tuning?** RAG = kasih data lewat prompt saat runtime, murah & data gampang di-update.
  Fine-tuning = latih ulang model, untuk mengubah gaya/perilaku, lebih mahal.
- **Embedding itu apa?** Teks diubah jadi deretan angka (vektor) yang mewakili makna.
  Teks yang maknanya mirip → vektornya berdekatan (diukur pakai cosine similarity).
- **Kenapa tool calling, bukan masukkan semua data ke prompt?** Data order/stok berubah real-time
  dan besar; tool mengambil hanya yang dibutuhkan, lebih akurat & hemat token.
- **Kenapa pakai framework (LangChain dll) atau tidak?** Di prototipe ini saya tulis loop agent
  manual supaya paham cara kerjanya dari dasar. Untuk sistem lebih besar, framework seperti
  LangChain/CrewAI mempercepat — dan karena paham dasarnya, saya bisa pakai framework dengan benar.
- **Pengalaman AI kamu?** Jujur: *"Pengalaman production saya masih terbatas. Tapi saya belajar
  cepat — prototipe ini contohnya: saya pelajari RAG, tool calling, dan guardrails, lalu langsung
  terapkan ke kasus PREPARE. Saya lebih suka belajar dengan membangun sesuatu yang nyata."*

## 8. Balik bertanya ke mereka
- "AI yang sedang atau ingin dibangun sekarang untuk apa — CS, marketing, atau internal?"
- "Datanya sekarang ada di mana — marketplace, spreadsheet, atau database sendiri?"
- "Tim AI-nya sudah ada berapa orang, dan pakai model apa sekarang?"
