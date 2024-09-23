python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_svhn\vgg11_cbnn8_16.py" --work-dir "D:\study\S3\Project thesis\Results\2D classification\8_16\vgg11_cbnn" --gpus 1
echo "Finished vgg11_cbnn8_16!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_minist\mobilenetv2_cbnn8_10.py" --work-dir "D:\study\S3\Project thesis\Results\2D classification\8_10\mobilenetv2_cbnn" --gpus 1
echo "Finished mobilenetv2_cbnn8_10!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_minist\mobilenetv2_cbnn8_14.py" --work-dir "D:\study\S3\Project thesis\Results\2D classification\8_14\mobilenetv2_cbnn" --gpus 1
echo "Finished mobilenetv2_cbnn8_14!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_minist\mobilenetv2_cbnn8_16.py" --work-dir "D:\study\S3\Project thesis\Results\2D classification\8_16\mobilenetv2_cbnn" --gpus 1
echo "Finished mobilenetv2_cbnn8_16!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_cifar10\resnet34_cbnn8_10_adam_1e-4_cosinelr.py" --work-dir "D:\study\S3\Project thesis\Results\2D classification\8_10\resnet34_cbnn" --gpus 1
echo "Finished resnet34_cbnn8_10!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_cifar10\resnet34_cbnn8_14_adam_1e-4_cosinelr.py" --work-dir "D:\study\S3\Project thesis\Results\2D classification\8_14\resnet34_cbnn" --gpus 1
echo "Finished resnet34_cbnn8_14!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_cifar10\resnet34_cbnn8_16_adam_1e-4_cosinelr.py" --work-dir "D:\study\S3\Project thesis\Results\2D classification\8_16\resnet34_cbnn" --gpus 1
echo "Finished resnet34_cbnn8_16!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_cbnn8_10.py" --work-dir "D:\study\S3\Project thesis\Results\Speech detection\dfsmn_cbnn\8_10" --gpus 1
echo "Finished dfsmn_cbnn8_10!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_cbnn8_14.py" --work-dir "D:\study\S3\Project thesis\Results\Speech detection\dfsmn_cbnn\8_14" --gpus 1
echo "Finished dfsmn_cbnn8_14!"
python tools\train.py "D:\study\S3\Project thesis\Work\BiBenchExt\BiBench\configs\acc_speech\dfsmn_cbnn8_16.py" --work-dir "D:\study\S3\Project thesis\Results\Speech detection\dfsmn_cbnn\8_16" --gpus 1
echo "Finished dfsmn_cbnn8_16!"
pause