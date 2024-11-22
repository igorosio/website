from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import aiplatform
import json
import base64

app = Flask(__name__)

# Inizializza i client Google Cloud
speech_client = speech.SpeechClient()
tts_client = texttospeech.TextToSpeechClient()
aiplatform.init(project="your-project-id", location="your-location")  # Sostituisci con i tuoi valori

# Simulazione dello stato degli scomparti
scomparti = {"scomparto_1": "libero", "scomparto_2": "occupato"}

# Tariffe
tariffe = {
    "piccolo": {"base": 1.50, "ora": 0.50},
    "medio": {"base": 2.50, "ora": 0.75},
    "grande": {"base": 3.50, "ora": 1.00}
}

# Codici promozionali
codici_promo = {
    "SCONTO20": 0.20,
    "SCONTO50": 0.50
}

def riconosci_voce(audio_bytes):
  """Riconosce la voce dall'audio e la converte in testo."""
  audio = speech.RecognitionAudio(content=audio_bytes)
  config = speech.RecognitionConfig(
      encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
      sample_rate_hertz=44100,
      language_code="it-IT",
  )
  response = speech_client.recognize(config=config, audio=audio)
  try:
    testo = response.results[0].alternatives[0].transcript
    print("Hai detto: " + testo)
    return testo
  except IndexError:
    print("Non ho capito.")
    return ""

def elabora_con_gemini(testo):
  """Invia il testo all'API Gemini e interpreta la risposta."""
  prompt = f"""
  L'utente ha detto: {testo}. 

  Stato degli scomparti: {scomparti}
  Tariffe: {tariffe}
  Codici promozionali validi: {codici_promo}

  I possibili intent sono:
  * aprire_scomparto
  * applicare_codice_promo
  * controllare_tariffa

  Rispondi con un JSON nel seguente formato:
  {{
    "intent": "intent_rilevato",
    "entità": {{
      "codice_promo": "codice_promozionale", 
      "dimensione_box": "dimensione_box",
      "numero_scomparto": "numero_scomparto" 
    }},
    "risposta": "risposta_testuale" 
  }}
  """
  response = aiplatform.predict(
      model="gemini-pro", 
      prompt=prompt, 
      max_output_tokens=1024
  )
  # Estrai l'intent, le entità e la risposta dal JSON
  try:
    response_json = json.loads(response.predictions[0].output)
    intent = response_json["intent"]
    entità = response_json["entità"]
    risposta = response_json["risposta"]
    return intent, entità, risposta
  except (json.JSONDecodeError, KeyError) as e:
    print(f"Errore nell'elaborazione della risposta di Gemini: {e}")
    return None, None, "Si è verificato un errore."

def apri_box(numero_scomparto):
  """Simula l'apertura del box."""
  if scomparti.get(numero_scomparto) == "libero":
    scomparti[numero_scomparto] = "aperto"
    return f"Scomparto {numero_scomparto} aperto."
  else:
    return f"Scomparto {numero_scomparto} non disponibile."

def esegui_azione(intent, entità, risposta):
  """Esegue l'azione in base all'intent."""
  if intent == "aprire_scomparto":
    numero_scomparto = entità.get("numero_scomparto")
    if numero_scomparto:
      risposta = apri_box(numero_scomparto)
    else:
      risposta = "Numero di scomparto non specificato."
  elif intent == "applicare_codice_promo":
    codice = entità.get("codice_promo")
    if codice in codici_promo:
      sconto = codici_promo[codice]
      risposta = f"Codice promozionale {codice} applicato! Sconto del {sconto*100}%."
    else:
      risposta = "Codice promozionale non valido."
  elif intent == "controllare_tariffa":
    dimensione = entità.get("dimensione_box")
    if dimensione in tariffe:
      base = tariffe[dimensione]["base"]
      ora = tariffe[dimensione]["ora"]
      risposta = f"Tariffa per un box {dimensione}: {base} euro di base + {ora} euro all'ora."
    else:
      risposta = "Dimensione del box non valida."
  return risposta

def genera_risposta_vocale(testo):
  """Genera una risposta vocale dal testo e la codifica in base64."""
  synthesis_input = texttospeech.SynthesisInput(text=testo)
  response = tts_client.synthesize_speech(
      input=synthesis_input, voice=texttospeech.VoiceSelectionParams(
          language_code="it-IT", name="it-IT-Wavenet-D"
      ), audio_config=texttospeech.AudioConfig(
          audio_encoding=texttospeech.AudioEncoding.MP3
      )
  )
  # Codifica l'audio in base64
  audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
  return f"data:audio/mp3;base64,{audio_base64}"

@app.route("/", methods=["GET", "POST"])
def index():
  if request.method == "POST":
    # Ottieni l'audio registrato dal microfono (in formato base64)
    audio_data = request.form.get("audio_data") 
    if audio_data:
      # Decodifica l'audio da base64
      audio_bytes = base64.b64decode(audio_data.split(",")[1])

      # Riconosci la voce dall'audio
      testo_utente = riconosci_voce(audio_bytes)

      # Elabora la richiesta con Gemini
      intent, entità, risposta_gemini = elabora_con_gemini(testo_utente)

      # Esegui l'azione appropriata
      risposta_testuale = esegui_azione(intent, entità, risposta_gemini)

      # Genera la risposta vocale
      audio_risposta_base64 = genera_risposta_vocale(risposta_testuale)

      # Restituisci la risposta come JSON
      return jsonify({"risposta": risposta_testuale, "audio_risposta": audio_risposta_base64}) 
    else:
      return jsonify({"errore": "Nessun dato audio ricevuto."})
  else:
    return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0")
