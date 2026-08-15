import os
import shutil

source_dir = "data/raw"          # folder where everything is currently mixed
images_out = "data/all_images"
labels_out = "data/all_labels"

# os.makedirs(images_out, exist_ok=True)
# os.makedirs(labels_out, exist_ok=True)

# moved_images, moved_labels, skipped = 0, 0, []

# for fname in os.listdir(source_dir):
#     full_path = os.path.join(source_dir, fname)
#     if not os.path.isfile(full_path):
#         continue

#     if fname.lower().endswith(".jpg") or fname.lower().endswith(".jpeg"):
#         shutil.copy(full_path, os.path.join(images_out, fname))
#         moved_images += 1
#     elif fname.endswith(".txt"):
#         shutil.copy(full_path, os.path.join(labels_out, fname))
#         moved_labels += 1
#     else:
#         skipped.append(fname)

# print(f"Moved {moved_images} images, {moved_labels} labels.")
# print(f"Skipped (unexpected file type): {skipped}")

# Verifing that every image has a matching label

images = {f.rsplit(".", 1)[0] for f in os.listdir(images_out)}
labels = {f.rsplit(".", 1)[0] for f in os.listdir(labels_out)}

missing_labels = images - labels   # images with no txt file
missing_images = labels - images   # txt files with no matching image

print(f"Images missing labels: {len(missing_labels)}")
print(f"Labels missing images: {len(missing_images)}")

if missing_labels:
    print("Example:", list(missing_labels)[:5])
if missing_images:
    print("Example:", list(missing_images)[:5])


prefix_to_class = {
    "speedbreaker": 0,   # adjust these prefixes to match your actual filenames
    "pothole": 1,
    "unpaved": 2,
}

mismatches = []
for fname in os.listdir(labels_out):
    name_lower = fname.lower()
    expected_class = None
    for prefix, cls_id in prefix_to_class.items():
        if prefix in name_lower:
            expected_class = cls_id
            break

    if expected_class is None:
        continue  # filename didn't match any known prefix, skip check

    with open(os.path.join(labels_out, fname)) as f:
        for line in f:
            actual_class = int(line.split()[0])
            if actual_class != expected_class:
                mismatches.append((fname, actual_class, expected_class))

print(f"Found {len(mismatches)} mismatches")
print(mismatches[:10])


from collections import Counter

confusion = Counter()   # (expected_class, actual_class) -> count

for fname in os.listdir(labels_out):
    name_lower = fname.lower()
    expected_class = None
    for prefix, cls_id in prefix_to_class.items():
        if prefix in name_lower:
            expected_class = cls_id
            break
    if expected_class is None:
        continue

    with open(os.path.join(labels_out, fname)) as f:
        for line in f:
            actual_class = int(line.split()[0])
            confusion[(expected_class, actual_class)] += 1

for (exp, act), count in sorted(confusion.items(), key=lambda x: -x[1]):
    print(f"Expected {exp}, Actual {act}: {count} lines")

import os

sample_files = os.listdir(labels_out)[:4000]
prefixes_found = set()
for f in sample_files:
    # print the part before the last underscore+number
    prefixes_found.add(f.rsplit("_", 1)[0])

print(prefixes_found)

import re

class_keywords = {
    0: ["sb_", "speed_", "unmarkedbump"],       # speed breaker
    1: ["pothole", "pot_holes"],                 # pothole
    2: ["unpaved", "ungradedroad", "unpavedroad"], # unpaved road
}

def get_expected_classes(fname):
    """Returns a set of expected class ids based on filename, or None if unrecognized."""
    name_lower = fname.lower()

    if name_lower.startswith("frame+") or name_lower.startswith("frame_"):
        return None   # no class info in filename — can't check these

    found = set()
    for cls_id, keywords in class_keywords.items():
        if any(kw in name_lower for kw in keywords):
            found.add(cls_id)

    return found if found else None


from collections import Counter

confusion = Counter()
unrecognized_files = []

for fname in os.listdir(labels_out):
    expected = get_expected_classes(fname)
    if expected is None:
        unrecognized_files.append(fname)
        continue

    with open(os.path.join(labels_out, fname)) as f:
        for line in f:
            actual_class = int(line.split()[0])
            if actual_class not in expected:
                confusion[(tuple(sorted(expected)), actual_class)] += 1

print(f"Unrecognized (frame+ etc.) files: {len(unrecognized_files)}")
for (exp, act), count in sorted(confusion.items(), key=lambda x: -x[1]):
    print(f"Expected {exp}, Actual {act}: {count} lines")

import re
from collections import Counter, defaultdict

def get_source_tag(fname):
    """Extract the naming pattern before the trailing number, to identify the source dataset."""
    name = fname.rsplit(".", 1)[0]
    # strip trailing _<number> or +<number>
    tag = re.sub(r'[_+]\d+$', '', name)
    return tag

source_class_dist = defaultdict(Counter)

for fname in os.listdir(labels_out):
    tag = get_source_tag(fname)
    with open(os.path.join(labels_out, fname)) as f:
        for line in f:
            cls_id = int(line.split()[0])
            source_class_dist[tag][cls_id] += 1

for tag, dist in sorted(source_class_dist.items()):
    total = sum(dist.values())
    print(f"{tag} (total {total} lines): {dict(dist)}")

import os

def fix_an_unpaved_labels(labels_dir):
    fixed_count = 0
    for fname in os.listdir(labels_dir):
        if not fname.lower().startswith("an_unpaved"):
            continue

        path = os.path.join(labels_dir, fname)
        new_lines = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id == 1:
                    cls_id = 2   # remap: their "1" = unpaved -> our "2"
                new_lines.append(f"{cls_id} {' '.join(parts[1:])}")
                fixed_count += 1

        with open(path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

    print(f"Remapped {fixed_count} lines across AN_unpaved files")

fix_an_unpaved_labels(labels_out)