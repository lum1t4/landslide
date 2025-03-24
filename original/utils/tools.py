import numpy as np
import matplotlib.pyplot as plt

def eval_image(predict,label,num_classes):
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)

        # Stampa delle etichette e delle previsioni
        # print("Labels:")
        # print(label)
        # print("Predictions:")
        # print(predict)

        # Visualizza le etichette e le previsioni come immagini
        # plt.figure(figsize=(10, 5))
        #
        # plt.subplot(1, 2, 1)
        # plt.title('Labels')
        # plt.imshow(label.reshape((int(np.sqrt(len(label))), int(np.sqrt(len(label))))), cmap='gray')
        # plt.axis('off')
        #
        # plt.subplot(1, 2, 2)
        # plt.title('Predictions')
        # plt.imshow(predict.reshape((int(np.sqrt(len(predict))), int(np.sqrt(len(predict))))), cmap='gray')
        # plt.axis('off')
        #
        # plt.show()



    return TP,FP,TN,FN,len(label)


