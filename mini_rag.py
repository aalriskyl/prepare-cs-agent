import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

def read_document():
    # Membaca isi file txt ke dalam variabel
    with open("company_info.txt", "r", encoding="utf-8") as file:
        return file.read()

def main():
    print("===========================================")
    print("🤖 Chatbot SOP HRD (Konsep Dasar RAG)")
    print("===========================================")
    
    # 1. RETRIEVAL: Mengambil data perusahaan
    company_data = read_document()
    
    print("Ketik 'keluar' untuk mengakhiri obrolan.\n")
    
    while True:
        pertanyaan = input("Kamu: ")
        
        if pertanyaan.lower() == 'keluar':
            print("Bot: Sampai jumpa!")
            break
            
        # 2. AUGMENTED GENERATION: Menggabungkan data kita ke dalam Prompt AI
        prompt = f"""
        Kamu adalah asisten HRD virtual yang ramah. 
        Tugasmu adalah menjawab pertanyaan karyawan HANYA berdasarkan teks [INFORMASI PERUSAHAAN] di bawah ini.
        Jika jawabannya tidak ada di dalam teks, jawab dengan "Maaf, informasi tersebut tidak ada di dalam dokumen SOP."
        Jangan mengarang jawaban (no hallucination).
        
        [INFORMASI PERUSAHAAN mulia]
        {company_data}
        [INFORMASI PERUSAHAAN selesai]
        
        [PERTANYAAN KARYAWAN]:
        {pertanyaan}
        """
        
        try:
            # Mengirim prompt ke Gemini
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            print(f"> Bot HRD: {response.text}\n")
            
        except Exception as e:
            print(f"[!] Error: {e}\n")

if __name__ == "__main__":
    main()
