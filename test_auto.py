import os
from core.oto_generator import generate_oto

# Create dummy folders
os.makedirs("test_wavs/textgrids", exist_ok=True)

# We need a dummy textgrid file to test
import textgrid
tg = textgrid.TextGrid()
words = textgrid.IntervalTier(name="words")
words.add(0.0, 0.5, "가")
words.add(0.5, 1.0, "나")
tg.append(words)

phones = textgrid.IntervalTier(name="phones")
phones.add(0.0, 0.1, "k")
phones.add(0.1, 0.5, "a")
phones.add(0.5, 0.6, "n")
phones.add(0.6, 1.0, "a")
tg.append(phones)

tg.write("test_wavs/textgrids/test.TextGrid")

# Open a dummy wav
with open("test_wavs/test.wav", "wb") as f:
    f.write(b"dummy")

def callback(msg):
    print(msg)

proc, tot, errs = generate_oto(
    tg_folder="test_wavs/textgrids",
    tpl_path="",
    out_path="test_wavs/oto_out.ini",
    fallback_format='vcv',
    callback=callback
)

print(f"Processed: {proc}/{tot}")
if errs:
    print("Errors:", errs)

if os.path.exists("test_wavs/oto_out.ini"):
    with open("test_wavs/oto_out.ini", "r", encoding="utf-8") as f:
        print("--- OTO.INI ---")
        print(f.read())
