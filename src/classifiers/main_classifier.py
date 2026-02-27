from clf import main_classify

#regions = ['all_data_paths', 'apk_full_memory', 'apk_full', 'apk_full_stack', 'base_apk_memory', 'base_apk', 'base_apk_stack', 'complete_memory_regions', 'data_stack', 'stack']
regions = ['data_stack', 'apk_full_stack']
sonify = ['wav8']

for r in regions:
    for s in sonify:
        print(f"Analysing {r} with {s}")
        main_classify(r, s)