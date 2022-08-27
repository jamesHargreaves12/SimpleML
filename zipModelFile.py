import pathlib
import shutil
import os
import py7zr


def z7_compress(folderpath):
    path = pathlib.Path(folderpath)
    with py7zr.SevenZipFile(os.path.join(path.parent,path.name+"_archive.7z"), 'w') as archive:
        archive.writeall(path,path.name)

    shutil.rmtree(folderpath)


if __name__ == "__main__":
    for path, subdirs, files in os.walk(r"C:\Users\james.hargreaves\PycharmProjects\SimpleMLRepo\JobOrchestrationWorkspace\Output_Checkpoint2"):
        if "model" in subdirs:
            z7_compress(os.path.join(path, "model"))
