import os 

def find_db3_files(directory):
    db3_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.db3'):
                db3_files.append(os.path.join(root, file))
    return db3_files