from transformers import pipeline

depth_estimator = pipeline(task="depth-estimation")
preds = depth_estimator("res/pedestrians-crosswalk.jpg")
print(preds)
preds["depth"].save("res/depth.jpeg")