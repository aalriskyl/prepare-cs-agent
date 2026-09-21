import os
from dotenv import load_dotenv
from google import genai

# Load variabel dari file .env
load_dotenv()

# Inisialisasi client Gemini
# Otomatis akan mengambil GEMINI_API_KEY dari .env
client = genai.Client()

def main():
    print("Menghubungi Gemini...")
    
    try:
        # Membuat request sederhana ke model gemini-3.6-flash
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents="Hai! Berikan saya satu kalimat motivasi pendek untuk interview kerja besok."
        )
        
        # Mengambil jawaban dari response
        ai_message = response.text
        
        print("\nJawaban AI:")
        print(f"> {ai_message}")
        
    except Exception as e:
        print("\n[!] Terjadi kesalahan!")
        print("Pastikan kamu sudah memasukkan GEMINI_API_KEY yang valid di file .env")
        print(f"Detail error: {e}")

if __name__ == "__main__":
    main()
