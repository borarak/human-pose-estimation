import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, UpSampling2D


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, input_shape=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters=128,
                            kernel_size=1,
                            padding='same',
                            strides=1,
                            name="res_cv1",
                            kernel_initializer='random_uniform')
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()
        self.conv2 = Conv2D(filters=128,
                            kernel_size=3,
                            padding='same',
                            strides=1,
                            name="res_cv2",
                            kernel_initializer='random_uniform')
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()
        self.conv3 = Conv2D(filters=256,
                            kernel_size=1,
                            padding='same',
                            strides=1,
                            name="res_cv3",
                            kernel_initializer='random_uniform')
        self.bn3 = BatchNormalization()
        self.relu3 = ReLU()

    def call(self, inputs):
        original_inp = inputs
        x = self.relu1(self.bn1(self.conv1(inputs)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return tf.add(original_inp, x)


class HourGlass():
    def __init__(self, depth=4):
        super(HourGlass, self).__init__()
        self.depth = depth
        self.down_res_blocks = [ResidualBlock() for _ in range(self.depth)]
        self.skip_block = [ResidualBlock() for _ in range(self.depth)]
        self.mpool_block = [MaxPool2D(strides=2) for _ in range(self.depth)]
        self.up_res_blocks = [ResidualBlock() for _ in range(self.depth)]
        self.upsample_block = [
            UpSampling2D(size=(2, 2)) for _ in range(self.depth)
        ]

        down_x = []

        inputs = tf.keras.layers.Input(shape=(64, 64, 256), name='hg')
        x = None

        for i in range(self.depth):
            if x is None:
                x = self.down_res_blocks[i](inputs)
            else:
                x = self.down_res_blocks[i](x)
            down_x.append(self.skip_block[i](x))
            x = self.mpool_block[i](x)

        down_x = list(reversed(down_x))

        for i in range(self.depth):
            x = self.up_res_blocks[i](x)
            x = self.upsample_block[i](x)
            x = tf.add(x, down_x[i])

        self.model = tf.keras.models.Model(inputs, x)


class HGNet():
    def __init__(self, stacks=1, num_keypoints=17):
        super(HGNet, self).__init__()
        self.stacks = stacks
        self.num_keypoints = num_keypoints
        self.hg = [HourGlass().model for _ in range(self.stacks)]
        self.final_res = [ResidualBlock() for _ in range(self.stacks)]
        self.final_res2 = [ResidualBlock() for _ in range(self.stacks - 1)]
        self.to_logits = [
            Conv2D(filters=self.num_keypoints,
                   kernel_size=1,
                   padding='same',
                   activation='sigmoid',
                   kernel_initializer='random_uniform')
            for _ in range(self.stacks)
        ]
        self.logits2inp = [
            Conv2D(filters=256,
                   kernel_size=1,
                   padding='same',
                   kernel_initializer='random_uniform')
            for _ in range(self.stacks - 1)
        ]
        self.outputs = []

        input = tf.keras.layers.Input(shape=(256, 256, 3))
        x = Conv2D(filters=256,
                   kernel_size=7,
                   strides=2,
                   padding='same',
                   kernel_initializer='random_uniform')(input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = ResidualBlock()(x)
        x = MaxPool2D(strides=2)(x)
        x = ResidualBlock()(x)
        hg_input = ResidualBlock()(x)

        for stack in range(self.stacks):
            x = self.hg[stack](hg_input)
            x = self.final_res[stack](x)
            logit = self.to_logits[stack](x)
            if stack < self.stacks - 1:
                x = self.final_res2[stack](x)
                inputs = self.logits2inp[stack](logit)
                hg_input = tf.add_n([hg_input, x, inputs])
            self.outputs.append(logit)

        self.model = tf.keras.models.Model(input, self.outputs)
