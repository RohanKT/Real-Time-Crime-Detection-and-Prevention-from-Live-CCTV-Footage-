from __future__ import division
from flask import Flask, jsonify, render_template,Response,request
import cv2
import os
import time
#os.add_dll_directory(r"C:\Users\Pratheek SB\AppData\Local\Programs\Python\Python38\Lib\site-packages\mxnet")
import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms import video
from gluoncv.data import VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory
import pickle    

import gc
from gluoncv.utils.filesystem import try_import_decord
from turbo_flask import Turbo
from gluoncv.data import Kinetics400Attr, UCF101Attr, SomethingSomethingV2Attr, HMDB51Attr
ansarr=[]
global final_pred_label
final_pred_label="N/A"
global stop
stop=0
def classifierSF():
    num_gpus = 1
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    per_device_batch_size = 4
    num_workers = 0
    batch_size = per_device_batch_size * num_gpus
    transform_train = transforms.Compose([
        video.VideoMultiScaleCrop(size=(224, 224), scale_ratios=[1.0, 0.875, 0.75, 0.66]),
        video.VideoRandomHorizontalFlip(),
        video.VideoToTensor(),
        video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = VideoClsCustom(root=r"Input",
                                setting=r"train.txt",
                                train=True,
                                slowfast=True,
                                new_length=64,
                                video_loader=True,
                                use_decord=True,
                                transform=transform_train)
    print('Load %d training samples.' % len(train_dataset))
    train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)

    net = get_model(name='slowfast_4x16_resnet50_custom', nclass=2)
    net.collect_params().reset_ctx(ctx)
    print(net)

    num_batches = len(train_data)

    steps_epochs = [4, 7, 9]
    # assuming we keep partial batches, see `last_batch` parameter of DataLoader
    iterations_per_epoch = math.ceil(len(train_dataset) / batch_size)
    # iterations just before starts of epochs (iterations are 1-indexed)
    steps_iterations = [s*iterations_per_epoch for s in steps_epochs]
    print("Learning rate drops after iterations: {}".format(steps_iterations))

    lr_schedule = mx.lr_scheduler.MultiFactorScheduler(step=steps_iterations, factor=0.1)

    optimizer = mx.optimizer.SGD(learning_rate=0.03, lr_scheduler=lr_schedule)

    trainer = gluon.Trainer(net.collect_params(), optimizer)
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    train_metric = mx.metric.Accuracy()
    train_history = TrainingHistory(['training-acc'])

    epochs = 100

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        train_loss = 0

        for i, batch in enumerate(train_data):
            data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            with ag.record():
                output = []
                for _, X in enumerate(data):
                    X = X.reshape((-1,) + X.shape[2:])
                    pred = net(X)
                    output.append(pred)
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

            for l in loss:
                l.backward()

            trainer.step(batch_size)

            train_loss += sum([l.mean().asscalar() for l in loss])
            train_metric.update(label, output)

            if i == 10:
                break

        name, acc = train_metric.get()

        train_history.update([acc])
        print('[Epoch %d] train=%f loss=%f time: %f' %
            (epoch, acc, train_loss / (i+1), time.time()-tic))

    train_history.plot()

    net.save_parameters('net.params')
    print("--------------------------------------------------------------------------\n\n The network is:",net)
    print("\n\n--------------------------------------------------------------------------")
    pickle.dump(net, open('trained_model.sav', 'wb'))
    return net



def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions on your own videos.')
    parser.add_argument('--data-dir', type=str, default='',
                        help='the root path to your data')
    parser.add_argument('--need-root', action='store_true',
                        help='if set to True, --data-dir needs to be provided as the root path to find your videos.')
    parser.add_argument('--data-list', type=str, default='',
                        help='the list of your data. You can either provide complete path or relative path.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='number of gpus to use. Use -1 for CPU')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                        help='enable using pretrained model from GluonCV.')
    parser.add_argument('--hashtag', type=str, default='',
                        help='hashtag for pretrained models.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--new-height', type=int, default=256,
                        help='new height of the resize image. default is 256')
    parser.add_argument('--new-width', type=int, default=340,
                        help='new width of the resize image. default is 340')
    parser.add_argument('--new-length', type=int, default=32,
                        help='new length of video sequence. default is 32')
    parser.add_argument('--new-step', type=int, default=1,
                        help='new step to skip video sequence. default is 1')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='number of classes.')
    parser.add_argument('--ten-crop', action='store_true',
                        help='whether to use ten crop evaluation.')
    parser.add_argument('--three-crop', action='store_true',
                        help='whether to use three crop evaluation.')
    parser.add_argument('--video-loader', action='store_true', default=True,
                        help='if set to True, read videos directly instead of reading frames.')
    parser.add_argument('--use-decord', action='store_true', default=True,
                        help='if set to True, use Decord video loader to load data.')
    parser.add_argument('--slowfast', action='store_true',
                        help='if set to True, use data loader designed for SlowFast network.')
    parser.add_argument('--slow-temporal-stride', type=int, default=16,
                        help='the temporal stride for sparse sampling of video frames for slow branch in SlowFast network.')
    parser.add_argument('--fast-temporal-stride', type=int, default=2,
                        help='the temporal stride for sparse sampling of video frames for fast branch in SlowFast network.')
    parser.add_argument('--num-crop', type=int, default=1,
                        help='number of crops for each image. default is 1')
    parser.add_argument('--data-aug', type=str, default='v1',
                        help='different types of data augmentation pipelines. Supports v1, v2, v3 and v4.')
    parser.add_argument('--num-segments', type=int, default=1,
                        help='number of segments to evenly split the video.')
    parser.add_argument('--save-dir', type=str, default='./predictions',
                        help='directory of saved results')
    parser.add_argument('--logging-file', type=str, default='predictions.log',
                        help='name of predictions log file')
    parser.add_argument('--save-logits', action='store_true',
                        help='if set to True, save logits to .npy file for each video.')
    parser.add_argument('--save-preds', action='store_true',
                        help='if set to True, save predictions to .npy file for each video.')
    parser.add_argument('--load',
                        help='load a saved pickle model file.')
    opt = parser.parse_args()
    return opt

def read_data(opt, video_name, transform, video_utils):
    

    decord = try_import_decord()
    decord_vr = decord.VideoReader(video_name, width=opt.new_width, height=opt.new_height)
    duration = len(decord_vr)
    opt.skip_length = opt.new_length * opt.new_step
    segment_indices, skip_offsets = video_utils._sample_test_indices(duration)

    if opt.video_loader:
        if opt.slowfast:
            clip_input = video_utils._video_TSN_decord_slowfast_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
        else:
            clip_input = video_utils._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
    else:
        raise RuntimeError('We only support video-based inference.')
    clip_input = transform(clip_input)

    if opt.slowfast:
        sparse_sampels = len(clip_input) // (opt.num_segments * opt.num_crop)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (sparse_sampels, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
    else:
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (opt.new_length, 3, opt.input_size, opt.input_size))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))

    if opt.new_length == 1:
        clip_input = np.squeeze(clip_input, axis=2)  

    return nd.array(clip_input)

def detect():
    opt = parse_args()
    makedirs(opt.save_dir)
    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    gc.set_threshold(100, 5, 5)

    if opt.gpu_id == -1:
        context = mx.cpu()
    else:
        gpu_id = opt.gpu_id
        context = mx.gpu(gpu_id)

    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if opt.ten_crop:
        transform_test = transforms.Compose([
            video.VideoTenCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 10
    elif opt.three_crop:
        transform_test = transforms.Compose([
            video.VideoThreeCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 3
    else:
        transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=image_norm_mean, std=image_norm_std)
        opt.num_crop = 1

    if opt.use_pretrained and len(opt.hashtag) > 0:
        opt.use_pretrained = opt.hashtag
    classes = opt.num_classes
    if opt.load:
        net = pickle.load(open(opt.load, 'rb'))
    else:
        net = classifierSF()
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.resume_params != '' and not opt.use_pretrained:
        net.load_parameters(opt.resume_params, ctx=context)
        logger.info('Pre-trained model %s is successfully loaded.' % (opt.resume_params))
    else:
        logger.info('Slowfast Custom model is successfully loaded.')
    classes = None

    anno_file = opt.data_list
    f = open(anno_file, 'r')
    data_list = f.readlines()
    logger.info('Load %d video samples.' % len(data_list))

    video_utils = VideoClsCustom(root=opt.data_dir,
                                 setting=opt.data_list,
                                 num_segments=opt.num_segments,
                                 num_crop=opt.num_crop,
                                 new_length=opt.new_length,
                                 new_step=opt.new_step,
                                 new_width=opt.new_width,
                                 new_height=opt.new_height,
                                 video_loader=opt.video_loader,
                                 use_decord=opt.use_decord,
                                 slowfast=opt.slowfast,
                                 slow_temporal_stride=opt.slow_temporal_stride,
                                 fast_temporal_stride=opt.fast_temporal_stride,
                                 data_aug=opt.data_aug,
                                 lazy_init=True)

    start_time = time.time()
    with open("Output.txt", "w") as text_file:
        for vid, vline in enumerate(data_list):
            video_path = r"{}".format(vline.split()[0])
            print(video_path)
            video_name = video_path.split('/')[-1]
            if opt.need_root:
                video_path = os.path.join(opt.data_dir, video_path)
            video_data = read_data(opt, video_path, transform_test, video_utils)
            video_input = video_data.as_in_context(context)
            pred = net(video_input.astype(opt.dtype, copy=False))
            if opt.save_logits:
                logits_file = '%s_%s_logits.npy' % ( video_name)
                np.save(os.path.join(opt.save_dir, logits_file), pred.asnumpy())
            pred_label = np.argmax(pred.asnumpy())
            if opt.save_preds:
                preds_file = '%s_%s_preds.npy' % ( video_name)
                np.save(os.path.join(opt.save_dir, preds_file), pred_label)

            if classes:
                pred_label = classes[pred_label]
                
            if(pred_label==1):
                    logger.info("Crime detected!")
            else:
                    logger.info("No crime detected!")
            logger.info('%04d/%04d: %s is predicted to class %s' % (vid, len(data_list), video_name, pred_label))
            global ansarr
            ansarr.append("vid:"+str(video_name)+"pred:"+str(pred_label))
            text_file.write("%s 64 %s\n" % (video_name,pred_label))
    end_time = time.time()
    logger.info('Total inference time is %4.2f minutes' % ((end_time - start_time) / 60))

'''-------------------------------------------------------------------app.py--------------------------------------------------------------------------'''

camera = cv2.VideoCapture(0)
filename = 'live/video.avi'
fps= 30.0
res = '720p' 
frame_arr=[]
count=0
video_no=1
out=0
STD_DIMENSIONS =  {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}


def change_res(camera, width, height):
    camera.set(3, width)
    camera.set(4, height)

def get_dims(camera,res='720p'):
    width, height = STD_DIMENSIONS["720p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    change_res(camera, width, height)
    return width, height

'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''

def gen_frames(res): 
        folder_path=r'D:\Capstone_Project_Binary_Classifier\live'
        while True:
            global frame_arr; 
            global count
            global video_no
            global out
            global filename
            filename1=''+filename
            success, frame = camera.read()  # read the camera frame
            if not success:
                break 
            else:
                    frame_arr.append(frame)
                    #print(len(frame_arr))
                    if(count== 30 * 10 and stop==0):
                        filenamev = filename1.split('.')[0]+str(video_no)+'.'+filename1.split('.')[1]
                        video_no+=1
                        cap = cv2.VideoCapture(0)
                        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
                        out = cv2.VideoWriter(filenamev,fourcc, 30, (int(cap.get(3)),int(cap.get(4))))
                        for i in frame_arr:
                            out.write(i)
                        if(video_no==1 or video_no==2):
                            continue
                        else:   
                            filenamemod = filename1.split('.')[0]+str(video_no-2)+'.'+filename1.split('.')[1]
                            print("-----------------------------------------------------------------"+str(video_no)+"------------------------------------")
                            with open('livelist.txt', 'w') as f: 
                                f.write(folder_path+'\\'+filenamemod.split("/")[1]+" 64 "+" 1")
                                f.write("\n")
                            f.close()
                            pathlive='livelist.txt'
                            detectVid(pathlive)
                        count=0
                        frame_arr=[]
                    count+=1
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
'''---------------------------------------------------------------------------------------------------------------------------------------------------------'''

def detectVid(pathlive):
    opt = parse_args()
    makedirs(opt.save_dir)
    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)
    logger.propagate = False

    gc.set_threshold(100, 5, 5)

    if opt.gpu_id == -1:
        context = mx.cpu()
    else:
        gpu_id = opt.gpu_id
        context = mx.gpu(gpu_id)

    image_norm_mean = [0.485, 0.456, 0.406]
    image_norm_std = [0.229, 0.224, 0.225]
    if opt.ten_crop:
        transform_test = transforms.Compose([
            video.VideoTenCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 10
    elif opt.three_crop:
        transform_test = transforms.Compose([
            video.VideoThreeCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize(image_norm_mean, image_norm_std)
        ])
        opt.num_crop = 3
    else:
        transform_test = video.VideoGroupValTransform(size=opt.input_size, mean=image_norm_mean, std=image_norm_std)
        opt.num_crop = 1

    if opt.use_pretrained and len(opt.hashtag) > 0:
        opt.use_pretrained = opt.hashtag
    classes = opt.num_classes
    if opt.load:
        net = pickle.load(open(opt.load, 'rb'))
    else:
        net = classifierSF()
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.resume_params != '' and not opt.use_pretrained:
        net.load_parameters(opt.resume_params, ctx=context)
        logger.info('Pre-trained model %s is successfully loaded.' % (opt.resume_params))
    else:
        logger.info('Slowfast Custom model is successfully loaded.')
    classes = None

    anno_file = pathlive
    f = open(anno_file, 'r')
    data_list = f.readlines()
    logger.info('Load %d video samples.' % len(data_list))

    video_utils = VideoClsCustom(root=opt.data_dir,
                                 setting=opt.data_list,
                                 num_segments=opt.num_segments,
                                 num_crop=opt.num_crop,
                                 new_length=opt.new_length,
                                 new_step=opt.new_step,
                                 new_width=opt.new_width,
                                 new_height=opt.new_height,
                                 video_loader=opt.video_loader,
                                 use_decord=opt.use_decord,
                                 slowfast=opt.slowfast,
                                 slow_temporal_stride=opt.slow_temporal_stride,
                                 fast_temporal_stride=opt.fast_temporal_stride,
                                 data_aug=opt.data_aug,
                                 lazy_init=True)

    start_time = time.time()
    with open("Output.txt", "w") as text_file:
        for vid, vline in enumerate(data_list):
            video_path = r"{}".format(vline.split()[0])
            print(video_path)
            video_name = video_path.split('/')[-1]
            if opt.need_root:
                video_path = os.path.join(opt.data_dir, video_path)
            video_data = read_data(opt, video_path, transform_test, video_utils)
            video_input = video_data.as_in_context(context)
            pred = net(video_input.astype(opt.dtype, copy=False))
            if opt.save_logits:
                logits_file = '%s_%s_logits.npy' % ( video_name)
                np.save(os.path.join(opt.save_dir, logits_file), pred.asnumpy())
            pred_label = np.argmax(pred.asnumpy())
            if opt.save_preds:
                preds_file = '%s_%s_preds.npy' % ( video_name)
                np.save(os.path.join(opt.save_dir, preds_file), pred_label)

            if classes:
                pred_label = classes[pred_label]
                
            if(pred_label==1):
                    logger.info("Crime detected!")
            else:
                    logger.info("No crime detected!")
            global final_pred_label
            final_pred_label=str(pred_label)
            logger.info('%04d/%04d: %s is predicted to class %s' % (vid, len(data_list), video_name, pred_label))
            global ansarr
            ansarr.append("vid:"+str(video_name)+"pred:"+str(pred_label))
            text_file.write("%s 64 %s\n" % (video_name,pred_label))
    end_time = time.time()
    logger.info('Total inference time is %4.2f minutes' % ((end_time - start_time) / 60))

app= Flask(__name__)
turbo = Turbo(app)
@app.route("/home", methods=["GET", "POST"])
def index():
    return render_template("index.html")   

@app.route("/updates", methods=["GET"])
def updates():
    global stop
    if(stop==0):
        if(str(final_pred_label)=="N/A"):
            newvalue="Loading detection model..."
        elif(str(final_pred_label)==0):
            newvalue="Crime Detected!"
        else:
            newvalue="Normal behaviour."
    else:
        newvalue="Detection model is not running."
    return jsonify(detection=newvalue)


#background process happening without any refreshing
@app.route('/background_process_test')
def background_process_test():
    global stop
    stop=1
    print ("stop has been changed to ",stop)
    return ("nothing")  

@app.route('/background_process_test2')
def background_process_test2():
    global stop
    stop=0
    print ("stop has been changed to ",stop)
    return ("nothing")  

@app.route('/video_feed')
def video_feed():
    res ='720p'
    return Response(gen_frames(res), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    #detect()
    app.run()
