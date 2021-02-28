# Regression Report_109368011

Kaggle Competition page:[LINK](https://www.kaggle.com/c/machine-learning-ntut-2020-autumn-classification/leaderboard)
Rank(11/152):  
![rank](rank.png)

## 1.作法說明

本次作業在本地端完成
</br>
作業環境:
</br>
Ubuntu 18.04
</br>
Anaconda-Jupyter Notbook
</br>
Keras2.4.3
</br>
硬體設備:
</br>
CPU:I7 8700
</br>
GPU:GTX1070 Ti 8G

1. 本次作業沒有驗證集，在訓練模型時需自行從測試集分出部分資料當作驗證集。
</br>

2. 使用自行建構的簡單CNN模型來預測，詳細架構如下
    ```
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 64, 64, 64)        1792      
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 64, 256)       147712    
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 32, 32, 256)       0         
    _________________________________________________________________
    dropout (Dropout)            (None, 32, 32, 256)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 32, 32, 256)       590080    
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 32, 32, 256)       590080    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 32, 32, 256)       590080    
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 16, 16, 256)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 16, 256)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 16, 256)       590080    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 8, 8, 256)         0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 8, 8, 256)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 16384)             0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               8389120   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 20)                10260     
    =================================================================
    Total params: 10,909,204
    Trainable params: 10,909,204
    Non-trainable params: 0
    _________________________________________________________________
    ```
</br>

3. 最後將test data放進模型做預測，再將結果寫入csv檔後上傳kaggle集完成本次作業。
</br>

4. 最後上傳kaggle的public score最高為0.99663，private score為0.99711![Kaggle_score](kaggle_score.png)


## 2.程式方塊圖與寫法
程式方塊圖如下:
![程式方塊圖](workflow.png)
 - Import moudules
    ```py
    # 匯入相關所需的模組
    import os,sys
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    from random import shuffle
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    from sklearn.model_selection import train_test_split

    import tensorflow as tf
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint
    from keras.models import Sequential
    from keras.layers import *
    from keras.optimizers import *
    from keras import utils as np_utils

    def solve_cudnn_error():    #用來解決訓練時發生的cudnn錯誤
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    solve_cudnn_error() 
    ```
 - Load data
    將照片讀取出來，並縮放成64*64的大小來避免記憶體用量過高與加快訓練時間
    ```py

    def read_images_labels(path,i):
        for file in os.listdir(path):
            abs_path=os.path.abspath(os.path.join(path,file))
            if os.path.isdir(abs_path):
                i+=1
                temp=os.path.split(abs_path)[-1]
                name.append(temp)
                read_images_labels(abs_path,i)
                amount=int(len(os.listdir(path)))
                sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
            else:
                if file.endswith('.jpg'):
                    image = cv2.resize(cv2.imread(abs_path),(64,64))
                    images.append(image)
                    labels.append(i-1)
        return images, labels, name 

    def read_main(path):
        images, labels, name = read_images_labels(path,i=0)
        images=np.array(images,dtype=np.float32)/255
        labels=np_utils.to_categorical(labels, num_classes=20)
        np.savetxt('name.txt',name,delimiter=' ',fmt="%s")
        return images, labels

    images=[]
    labels=[]
    name=[]
    images,labels=read_main('simpsondata/train/characters-20')
    X_train,X_test,Y_train,Y_test=train_test_split(images,labels,test_size=0.2) #將資料拆分成訓練集與驗證集

    ```
    
- model construction

    模型建構如下

    ```py
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
        
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
        
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
        
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
        
    model.add(Dense(20, activation='softmax'))
        

    model.summary() # 秀出模型架構
    ```
    模型資訊如下

    ```py
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 64, 64, 64)        1792      
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 64, 64, 256)       147712    
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 32, 32, 256)       0         
    _________________________________________________________________
    dropout (Dropout)            (None, 32, 32, 256)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 32, 32, 256)       590080    
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 32, 32, 256)       590080    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 32, 32, 256)       590080    
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 16, 16, 256)       0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 16, 16, 256)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 16, 16, 256)       590080    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 8, 8, 256)         0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 8, 8, 256)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 16384)             0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               8389120   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 20)                10260     
    =================================================================
    Total params: 10,909,204
    Trainable params: 10,909,204
    Non-trainable params: 0
    _________________________________________________________________
    ```

    模型建置參數如下，這邊callback函數使用了checkpoint來將訓練過程中validat_accuracy最高的epoch的權重儲存起來。

    ```py
    model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
    callbacks_list= [checkpoint]
    ```

- traing model

    模型訓練如下

    ```py
    start=time.time()
    epochs=20
    batch_size=256
    history = model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            shuffle=True,verbose=1,callbacks=callbacks_list)
    ```
- validate model

   將訓練過程的loss與accuracy畫出來，觀察是否有under fitting或overfitting的現象

    ```py

    import matplotlib.pyplot as plt

    def plot_train_history(history, train_metrics, val_metrics):
        plt.plot(history.history.get(train_metrics),'-o')
        plt.plot(history.history.get(val_metrics),'-o')
        plt.ylabel(train_metrics)
        plt.xlabel('Epochs')
        plt.legend(['train', 'validation'])      
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plot_train_history(history, 'loss','val_loss')

    plt.subplot(1,2,2)
    plot_train_history(history, 'accuracy','val_accuracy')

    plt.show()
    ```
    ![訓練結果趨勢圖](training_curve.png)

- predicting answer

    將val_accuracy最高的weight load進model裡

    ```py
    model.load_weights("weights.best.hdf5")
    score=model.evaluate(X_test,Y_test,verbose=1)
    print(score) #確認val_accuracy是否與訓練時相同
    ```    
    讀取test data後丟進模型進行預測
    ```py
    import os 
    DIR = 'simpsondata/test/test' #test data的資料夾
    image_len=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) 
    
    def read_images(path):
        test_images=[]
        for i in range(image_len):
                img = cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(64,64))
                test_images.append(img)   
        test_images=np.array(test_images,dtype=np.float32)/255.
        return test_images
    test_img = read_images(DIR)
    pred = model.predict_classes(test_img,verbose=1)
    ```

- output Answer
    將預測結果轉回角色名稱
    ```py
    nametxt=np.loadtxt('name.txt',dtype='str')
    name_label=[]
    name_num=0
    for i in nametxt:
        name_label.append([name_num,i])
        name_num+=1
    new_pred=[]
    for i in range(len(pred)):
        new_pred.append(name_label[pred[i]][1])
    ```

    最後將預測結果寫入csv檔

    ```py
    with open('test1.csv','w+') as ans:
    ans.write('id,character\n')
    id=1
    for i in range(len(new_pred)):
        ans.write(str(id)+','+new_pred[i]+'\n')
        id+=1
    ```

- Convolution weight drawing
    Conv0
    ```py
    filters, biases = model.layers[0].get_weights()
    n_filters, ix = 64, 1
    plt.figure(figsize=(5,40))
    for i in range(n_filters):
        fj = filters[:, :, :, i]
        rgb= (np.dstack((fj[:,:,0],fj[:,:,1],fj[:,:,2])) * 255.999).astype(np.uint8)
        ax = plt.subplot(n_filters, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(rgb,cmap='hsv')
        ix += 1
    plt.show()
    ```
    Conv1~5
    ```py
    filters, biases = model.layers[1].get_weights() #change layer to get weight drawing of each conv layer
    n_filters, ix = 256, 1
    plt.figure(figsize=(10,200))
    for i in range(n_filters):
        fj = filters[:, :, :, i]
        rgb= (np.dstack((fj[:,:,0],fj[:,:,1],fj[:,:,2])) * 255.999).astype(np.uint8)
        ax = plt.subplot(n_filters, 16, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(rgb)
        ix += 1
    plt.show()
    ```
- Feature Map Drawing
    ```py
    from keras.models import Model
    from numpy import expand_dims
    ixs = [0,1,4,5,6,9]
    outputs = [model.layers[i].output for i in ixs]
    model_f = Model(inputs=model.inputs, outputs=outputs)
    img = X_train[1] 
    img = expand_dims(img, axis=0)
    feature_maps = model_f.predict(img)
    square = 8
    for fmap in feature_maps:
        ix = 1
        plt.figure(figsize=(10,10))
        for _ in range(square):
            for _ in range(square):
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(fmap[0, :, :, ix-1],cmap='gray')
                ix += 1
        plt.figure(figsize=(20,20))
        square=16
        plt.show()
    ```

- Confusion martix drawing
    
    ```py
    total_classcount=np.array([0 for _ in range(len(nametxt))]).astype(np.float)
    New_X_test=[]
    New_Y_test=[]
    endata=True
    while endata:
        endata=False
        w=np.random.randint(0,len(Y_test))
        for x in range(len(Y_test[w])):
            if int(Y_test[w][x]) > 0 :
                if total_classcount[x] < 20:
                    total_classcount[x]+=1
                    New_X_test.append(X_test[w])
                    New_Y_test.append(Y_test[w])
        for z in total_classcount:
            endata=True if z < 20 else endata
    New_X_test=np.array(New_X_test)
    New_Y_test=np.array(New_Y_test)
    print(New_X_test.shape)
    print(New_Y_test.shape)
    print(total_classcount)

    val_pred=model.predict_classes(New_X_test)

    Y_true=[]
    ta=New_Y_test
    for j in ta:
        c=0
        for i in j:
            if i==1.0:    
                Y_true.append(c)
                break
            else:
                c+=1
    nametxt,Y_true

    confusion_martix=np.array([[0 for _ in range(len(nametxt))] for _ in range(len(nametxt))]).astype(np.float)
    total_classcount=np.array([0 for _ in range(len(nametxt))]).astype(np.float)
    tp_martix=total_classcount
    for a in range(len(Y_true)):
        total_classcount[Y_true[a]]+=1.0
        confusion_martix[Y_true[a]][val_pred[a]]+=1.0

    import seaborn as sns
    plt.figure(figsize=(13,10))
    sns.heatmap(confusion_martix,cmap='Oranges',annot=True, xticklabels=nametxt,yticklabels=nametxt)
    plt.title("confusion_martix")
    ```
## 3.畫圖做結果分析

從validation data裡，每類隨機挑20張照片來畫confusion martix，結果如下，共有5張預測失誤
</br>
![confusion_martix](confusion_martix_orange.png)

本次作業使用的model共有6個卷積層，以下為各層的權重與feature的視覺化圖片

原圖
</br>
![source_pic](feature_map_source.png)
- conv0
    ```
    conv2d (Conv2D)     (None, 64, 64, 64)      
    ```
    weighet
    </br>
    ![conv0_weight](conv_filter_0.png)
    </br>
    feature map
    </br>
    ![conv0_feature_map](feature_map1.png)

- conv1
    ```
    conv2d_1 (Conv2D)       (None, 64, 64, 256)      
    ```
    weighet
    </br>
    ![conv1_weight](conv_filter_1.png)
    </br>
    feature map
    </br>
    ![conv1_feature_map](feature_map2.png)

- conv2
    ```
    conv2d_2 (Conv2D)       (None, 32, 32, 256)      
    ```
    weighet
    </br>
    ![conv2_weight](conv_filter_4.png)
    </br>
    feature map
    </br>
    ![conv2_feature_map](feature_map3.png)

- conv3
    ```
    conv2d_3 (Conv2D)       (None, 32, 32, 256)      
    ```
    weighet
    </br>
    ![conv3_weight](conv_filter_5.png)
    </br>
    feature map
    </br>
    ![conv3_feature_map](feature_map4.png)

- conv4
    ```
    conv2d_4 (Conv2D)       (None, 32, 32, 256)     
    ```
    weighet
    </br>
    ![conv4_weight](conv_filter_6.png)
    </br>
    feature map
    </br>
    ![conv4_feature_map](feature_map5.png)

- conv5
    ```
    conv2d_5 (Conv2D)       (None, 16, 16, 256)      
    ```
    weighet
    </br>
    ![conv5_weight](conv_filter_9.png)
    </br>
    feature map
    </br>
    ![conv5_feature_map](feature_map6.png)

## 4.討論預測值誤差很大的,是怎麼回事？
1. 本次作業的test data較為容易判斷，簡單的CNN模型即可取得不錯的成績，可能少部分的場景較為特殊或有其他角色混入，導致預測失誤。
</br>

2. 本次data set各角色的數量也有些失衡，可能也導致某些資料集較少的角色預測上較不準確。

## 5.如何改進

1. 在訓練圖像辨識的模型時，data set的數量與品質往往是影響最大的一環，data set若盡可能涵蓋到各種有機會出現的可能將能夠改善預測準度，常見的方法為data Agumentation，透過各種影像處理，如水平/垂直翻轉、裁切、旋轉、縮放、改變亮度/對比、增加雜訊......等等方式來盡可能增加資料集。
</br>

2. 目前已有許多強大的pre-train CNN model在市面上，這些model都已經經過非常多資料的訓練及調整，效能基本上都會比自己建構的CNN模型來的好，可以利用這些強大的model當作基礎並放入自己的資料級進行訓練。
</br>

3. 每個類別可能適合不同的model來進行預測，可嘗試用多種model來訓練，調查哪些情況使用哪一個模型最準確，並加入regressor來做輸出選擇，可以讓整個模型更加強大。
