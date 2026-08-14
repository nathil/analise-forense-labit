import os

class WorkspaceCleaner:
    TARGETS = {
        "data": ["data"],
        "results": ["results/pipelines", "results/reduction-analysis", "results/charts"],
        "charts": ["results/charts"],
        "multiple-pipelines": ["results/charts/multiple-pipelines"],
        "pipeline": ["results/charts/pipeline"],
        "pipelines": ["results/pipelines"],
        "reduction-analysis": ["results/reduction-analysis"]
    }

    @classmethod
    def clean(cls, targets: list[str]):
        """
        Deletes all files inside the specified logical target directories,
        except for .keep and __init__.py files.
        If 'all' is passed, cleans all known logical paths.
        """
        paths_to_clean = set()
        
        if "all" in [t.lower() for t in targets]:
            for v in cls.TARGETS.values():
                paths_to_clean.update(v)
        else:
            for t in targets:
                t_lower = t.lower()
                if t_lower in cls.TARGETS:
                    paths_to_clean.update(cls.TARGETS[t_lower])
                else:
                    print(f"[-] Warning: Unknown target '{t}'. Ignored.")

        deleted_count = 0
        for path in paths_to_clean:
            if not os.path.exists(path):
                continue
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file not in [".keep", "__init__.py"]:
                        file_path = os.path.join(root, file)
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            print(f"[-] Failed to delete {file_path}: {e}")
                            
        print(f"[+] Cleaned {deleted_count} files across selected targets.")
        return deleted_count
