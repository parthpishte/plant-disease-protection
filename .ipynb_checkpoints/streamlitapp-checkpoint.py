{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "30391891-589f-4359-bbdf-a10d401e58e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\parth\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import json\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import streamlit as st\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "3df48800-eea4-4fc2-b247-165d26bb11d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\parth\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\backend.py:1398: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.\n",
      "\n",
      "WARNING:tensorflow:From C:\\Users\\parth\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\layers\\pooling\\max_pooling2d.py:161: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "model_path='plantvillage dataset/model.h5'\n",
    "model=tf.keras.models.load_model(model_path)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1c118a39-caaa-47c6-b52d-0c6e471e4660",
   "metadata": {},
   "outputs": [],
   "source": [
    "class_indices=json.load(open('class_indices.json'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "be54d240-2a91-47d7-bdb0-420cb0f354b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_and_preprocess_iamge(image_path,target_size=(224,224)):\n",
    "    img=Image.open(image_path)\n",
    "    img=img.resize(target_size)\n",
    "    img_array=np.array(img)\n",
    "    img_array=np.expand.dims(img_array,axis=0)\n",
    "    img_array=img_array.astype('float32')/255\n",
    "    return img_array\n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d8dd39c-d6c6-4686-9a30-d448cde57419",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_img_class(model,image_path,class_indices):\n",
    "    preprocessed_img=load_and_preprocess_image(image_path)\n",
    "    predictions=model.predict(preprocessed_img)\n",
    "    predicted_class_index=np.argmax(predictions,axis=1)[0]\n",
    "    predicted_class_name=class_indices[predicted_class_index]\n",
    "    return predicted_class_name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cee8459c-6295-408e-b9f2-77e25f9427a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.title('Plant disease classifier')\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec94bdbb-90ff-4ef9-962c-78e24135e449",
   "metadata": {},
   "outputs": [],
   "source": [
    "uploaded_image=st.file_uploader('upload an image....',type=['jpg','jpeg','png'])\n",
    "if uploaded_image is not None:\n",
    "    image=Image.open(uploaded_image)\n",
    "    col1,col2=st.columns(2)\n",
    "\n",
    "    with col1:\n",
    "        resized_img=image.resize(150,150)\n",
    "        st.image(resized_img)\n",
    "\n",
    "    with col2:\n",
    "        if st.button(\"classify\"):\n",
    "            prediction=predict_img_class(model,uploaded_image,class_indices)\n",
    "            st.success(f'prediction:{str(prediction)}')\n",
    "            \n",
    "                                "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
