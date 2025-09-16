from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="multilingual_speech_recognition",device="gpu")

output = pipeline.predict("https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav")


for res in output:
    print(res)
    res.print() 
    res.save_to_json("./output/")  