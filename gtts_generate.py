# gtts_generate.py "<texto>" <output_mp3_path>
import sys
from gtts import gTTS

if len(sys.argv) < 3:
    print("usage: gtts_generate.py 'texto' out.mp3")
    sys.exit(1)

text = sys.argv[1]
out = sys.argv[2]
tts = gTTS(text, lang='pt')
tts.save(out)
print("saved")
