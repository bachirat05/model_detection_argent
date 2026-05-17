from roboflow import Roboflow
rf = Roboflow(api_key="eKSnQuHX9VrcmEF5R0Rw")
project = rf.workspace("aitoulahyans-workspace").project("pi-jsuku")
version = project.version(1)
dataset = version.download("yolov8")
                