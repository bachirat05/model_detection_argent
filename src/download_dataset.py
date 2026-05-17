from roboflow import Roboflow
rf = Roboflow(api_key="WdIzRcsGEDE0P5GVxyAs")
project = rf.workspace("nouraccess").project("money-detection-9kpvz")
version = project.version(1)
dataset = version.download("folder")