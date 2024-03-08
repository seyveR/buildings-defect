import ultralytics
from ultralytics import YOLO
import os
from PIL import Image
from ultralytics.engine.results import Results

model = YOLO(model="defseg/alldefectseg.pt")

image = Image.open('C:/Users/Vladimir/Downloads/Telegram Desktop/photoDiplom/photoDiplom/seams/defect/1SH_DEF_93_TIMME_2.jpg')
outputs: list[Results] = model.predict(image,
               show_labels=False,
               show_conf=False,
               save_txt=False,
               save_crop=False,
               save_conf=False,
               #    save=True
               )

for result in outputs:
    annot = result.plot(labels=True, conf=False, line_width=2)
    print(result.names[result.boxes.cls.item()])
    im = Image.fromarray(annot[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')