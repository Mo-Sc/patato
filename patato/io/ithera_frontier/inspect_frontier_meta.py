"""
Dump the .NET metadata of one Frontier capture so we can find sampling frequency,
speed of sound and transducer geometry.

Run on YOUR machine (needs .NET + your DB):
    python inspect_frontier_meta.py "C:\\path\\to\\db" patient 1 > meta_dump.txt
    python inspect_frontier_meta.py "C:\\path\\to\\db" calibration 2024-11-20T12:00:19 > meta_dump.txt
Then paste / attach meta_dump.txt.
"""
import sys
from custom_python_libraries import FrontierReader

db_path, source, session = sys.argv[1], sys.argv[2], sys.argv[3]
db = FrontierReader(db_path=db_path)._capture_db

if source == "patient":
    sess = db.GetPatient(int(session)).Result
else:
    sess = next(s for s in db.EnumCalibrations().Result
                if str(s.SessionStart.ToString("yyyy-MM-ddTHH:mm:ss")) == session)

cid = list(sess.EnumCaptures().Result)[0]
cap = sess.GetCapture(cid).Result
print("Capture:", cap.Name, cid.get_Guid().ToString())

SIMPLE = (int, float, str, bool)
seen = set()


def dump(obj, path, depth=0, max_depth=5):
    if depth > max_depth or id(obj) in seen:
        return
    seen.add(id(obj))
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            val = getattr(obj, name)
        except Exception as e:
            print(f"{path}.{name} = <error {type(e).__name__}>")
            continue
        if callable(val):
            continue
        if isinstance(val, SIMPLE) or val is None:
            print(f"{path}.{name} = {val!r}")
        elif hasattr(val, "__len__") and not hasattr(val, "GetType"):
            print(f"{path}.{name} = <sequence len={len(val)}>")
        else:
            print(f"{path}.{name}  [{type(val).__name__}]")
            dump(val, f"{path}.{name}", depth + 1, max_depth)


print("\n===== MasterMeta =====")
dump(cap.MasterMeta, "MasterMeta")

print("\n===== Tape =====")
dump(cap.Tape, "Tape", max_depth=2)

# Per-frame metadata of the first raw OA frame (sampling info, channels, etc.)
print("\n===== First frame meta =====")
try:
    frames = list(cap.Tape.GetFrames(0))
    for fr in frames[:3]:
        f = fr.GetFrame().Result
        print("--- frame type:", type(f).__name__)
        dump(f.Meta, "Frame.Meta", max_depth=3)
except Exception as e:
    print("could not read frame meta:", e)
