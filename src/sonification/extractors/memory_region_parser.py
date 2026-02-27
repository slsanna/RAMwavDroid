import os
import re
import logging

logger = logging.getLogger(__name__)

class MemoryRegionParser:
    def __init__(self):
        self.selection_map = {
            "stack": [r"\[stack\]"],
            "memory_regions": [r"\[stack\]", r"\[vvar\]"],
            "stack_vdso": [r"\[stack\]", r"\[vdso\]"],
            "base_odex-apk": [r"base\.odex", r"base\.apk"],
            "base_vdex-apk": [r"base\.vdex", r"base\.apk"],
            "base_odex-vdex": [r"base\.odex", r"base_vdex"],
            "base_odex": [r"base\.odex"],
            "base_vdex": [r"base\.vdex"],
            "base_apk": [r"base\.apk"],
            "complete_memory_regions": [r"\[heap\]", r"\[anon:libc_malloc\]", r"\[stack\]",
                r"\[vdso\]", r"\[vvar\]", r"\[vsyscall\]",
                r"\(deleted\)", r"u:object_r.*", r"data@resource-cache@"],
            "base_apk_memory": [r"\[stack\]", r"\[vvar\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
            "base_apk_stack": [r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex"],
            "all_data_paths": [r"^/data/.*"],
            "data_stack": [r"^/data/.*", r"\[stack\]"],
            "apk_full": [
                r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
                r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
                r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
                r"/data/.+\.odex$"
            ],
            "apk_full_memory": [r"\[stack\]", r"\[vvar\]",
                r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
                r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
                r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
                r"/data/.+\.odex$"
            ],
            "apk_full_stack": [r"\[stack\]", r"base\.apk", r"base\.vdex", r"base\.odex", r"/data/.*/lib/.+\.so$",
                r"\[anon:dalvik.*\]", r"\[anon:.*\.apk.*\]", r"\.db$", r"\.db-shm$", r"\.db-wal$",
                r"Cookies", r"Preferences", r"ua_preferences", r"data@resource-cache@", r"\.dex$",
                r"/data/.+\.odex$"],
            "system": [
                r"framework\.jar", r"boot\.vdex", r"boot-framework\.vdex", r"base\.odex"
            ],
            "system_full": [
                r"dalvik-classes", r"framework\.jar", r"boot-framework\.art",
                r"boot-core-libart\.art", r"boot\.oat", r"boot-framework\.oat",
                r"boot\.vdex", r"boot-framework\.vdex", r"base\.odex"
            ],
            "libraries": [r"\.so", r"\[anon:lib.*"]
        }

    def parse_maps_file(self, maps_path, selection_mode="all"):
        selected_regions = {}
        if not os.path.exists(maps_path):
            return selected_regions

        try:
            with open(maps_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 1:
                        continue

                    address = parts[0].split('-')[0].strip()
                    desc = line.split(None, 5)[-1] if len(line.split(None, 5)) == 6 else ""

                    if selection_mode in [None, "all"]:
                        selected_regions[address] = (desc, "all")
                    elif selection_mode in self.selection_map:
                        for pattern in self.selection_map[selection_mode]:
                            if re.search(pattern, desc):
                                selected_regions[address] = (desc, selection_mode)
                                break
        except Exception as e:
            logger.error(f"Failed to parse maps file {maps_path}: {e}")

        return selected_regions
