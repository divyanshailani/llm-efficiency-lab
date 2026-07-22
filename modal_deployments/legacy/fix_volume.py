import modal

app = modal.App("volume-fixer")
volume = modal.Volume.from_name("nvfp4-weights")

@app.function(volumes={"/root/weights": volume})
def fix():
    import os
    import shutil
    
    src = "/root/weights/deckard-40b-nvfp4_temp"
    dst = "/root/weights/deckard-40b-nvfp4"
    
    print(f"Checking for {src}...")
    if not os.path.exists(src):
        print(f"ERROR: {src} does not exist!")
        return
        
    print(f"Moving {src} -> {dst}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
        
    os.rename(src, dst)
    
    print("Committing volume...")
    volume.commit()
    print("SUCCESS: Volume fixed and committed.")
