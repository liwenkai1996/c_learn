paddle2onnx \
  --model_dir /root/.paddlex/official_models/PP-OCRv5_server_det \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file /root/.paddlex/official_models/PP-OCRv5_server_det/PP-OCRv5_server_det.onnx \
  --opset_version 18 \
  --enable_onnx_checker True


paddle2onnx \
  --model_dir /root/.paddlex/official_models/PP-LCNet_x1_0_textline_ori \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file /root/.paddlex/official_models/PP-LCNet_x1_0_textline_ori/PP-OCRv5_server_ori.onnx \
  --opset_version 18 \
  --enable_onnx_checker True

paddle2onnx \
  --model_dir /root/.paddlex/official_models/PP-OCRv5_server_rec \
  --model_filename inference.json \
  --params_filename inference.pdiparams \
  --save_file /root/.paddlex/official_models/PP-OCRv5_server_rec/PP-OCRv5_server_rec.onnx \
  --opset_version 18 \
  --enable_onnx_checker True






export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opt/local/tensorrt/targets/x86_64-linux-gnu/lib/ && cd /home/opt/local/tensorrt/bin
/home/opt/local/tensorrt/bin/trtexec \
  --onnx=/root/.paddlex/official_models/PP-OCRv5_server_det/PP-OCRv5_server_det.onnx \
  --saveEngine=/root/.paddlex/official_models/PP-OCRv5_server_det/PP-OCRv5_server_det.trt \
  --fp16 \
  --minShapes=x:1x3x1920x1920 \
  --optShapes=x:1x3x1920x1920 \
  --maxShapes=x:1x3x1920x1920 


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opt/local/tensorrt/targets/x86_64-linux-gnu/lib/ && cd /home/opt/local/tensorrt/bin
/home/opt/local/tensorrt/bin/trtexec \
  --onnx=/root/.paddlex/official_models/PP-LCNet_x1_0_textline_ori/PP-OCRv5_server_ori.onnx \
  --saveEngine=/root/.paddlex/official_models/PP-LCNet_x1_0_textline_ori/PP-OCRv5_server_ori.trt \
  --fp16 \
  --minShapes=x:1x3x80x160 \
  --optShapes=x:12x3x80x160 \
  --maxShapes=x:24x3x80x160 


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/opt/local/tensorrt/targets/x86_64-linux-gnu/lib/ && cd /home/opt/local/tensorrt/bin
/home/opt/local/tensorrt/bin/trtexec \
  --onnx=/root/.paddlex/official_models/PP-OCRv5_server_rec/PP-OCRv5_server_rec.onnx \
  --saveEngine=/root/.paddlex/official_models/PP-OCRv5_server_rec/PP-OCRv5_server_rec.trt \
  --fp16 \
  --minShapes=x:1x3x48x1920 \
  --optShapes=x:1x3x48x1920 \
  --maxShapes=x:1x3x48x1920 