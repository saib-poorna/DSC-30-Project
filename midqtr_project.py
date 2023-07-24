"""
DSC 20 Mid-Quarter Project
Name(s): Sai Poornasree Balamurugan, Catherine Back 
PID(s): A17561823, A17590624
"""

import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# Part 1: RGB Image #
class RGBImage:
    """
    TODO: add description
    """

    def __init__(self, pixels):
        """
        TODO: add description

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        if type(pixels) != list:
            raise TypeError
        elif len(pixels) == 0:
            raise TypeError
        
        for row in pixels:
            len_first_row = len(pixels[0])
            if type(row) != list:
                raise TypeError
            
            if len(row) == 0:
                raise TypeError
            
            if len_first_row != len(row):
                raise TypeError
            
            for pixs in row:
                if type(pixs) != list:
                    raise TypeError
                
                if len(pixs) != 3:
                    raise TypeError
                
                for val in pixs:
                    if val < 0 or val > 255:
                        raise ValueError
            
        
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])

    def size(self):
        """
        Returns the size of the image as a tuple

        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Returns the color of the pixel at a given postition as a tuple

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        return [[[val for val in pixs] for pixs in rows] for rows in self.pixels]
        
        

    def copy(self):
        """
        TODO: returns a COPY of the RGBImage instance

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        # YOUR CODE GOES HERE #
        return self.get_pixels()

    def get_pixel(self, row, col):
        """
        TODO: returns the color of the pixel at position (row, col)

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        # YOUR CODE GOES HERE #
        if (type(row) != int) or (type(col) != int):
            raise TypeError
        elif (row > self.num_rows - 1) or (col > self.num_cols - 1):
            raise ValueError
        elif row < 0 or col < 0:
            raise ValueError
         
        pixel = self.pixels[row][col]
        return tuple(pixel)

    def set_pixel(self, row, col, new_color):
        """
        TODO: updates the color of the pixel at position (row, col) to the new_color

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the resulting pixel list
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        # YOUR CODE GOES HERE #
        if type(row) != int or type(col) != int:
            raise TypeError
        elif (row > self.num_rows - 1) or (col > self.num_cols - 1):
            raise ValueError
        elif type(new_color) != tuple:
            raise TypeError
        elif len(new_color) != 3:
            raise TypeError
        for i in new_color:
            if type(i) != int:
                raise TypeError
            elif i > 255:
                raise ValueError

        updated_pixel = []
        get_curr_pixel = list(self.get_pixel(row,col))
        color_pixel = list(new_color)
        for i in range(len(get_curr_pixel)):
            if color_pixel[i] >= 0:
                updated_pixel.append(color_pixel[i])
            elif color_pixel[i] < 0:
                updated_pixel.append(get_curr_pixel[i])
        self.pixels[row][col] = updated_pixel


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        TODO: add description

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost 

    def negate(self, image):
        """
        TODO: returns the negative of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #
        RGB_list = image.get_pixels()
        negate_lst = [[[255 - val for val in col] for col in row] for row in RGB_list]
        
        negate_img = RGBImage(negate_lst)
        return negate_img

    

    def grayscale(self, image):
        """
        TODO: add description

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #
        pixel = image.get_pixels()
        gray_pixels = [[[sum(j)//3 for k in j] for j in i] for i in pixel]
        gray_img = RGBImage(gray_pixels)
        return gray_img
        
        
        

    def rotate_180(self, image):
        """
        TODO: add description

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        pixels = image.get_pixels()
        pixel_reverse = [[j for j in i][::-1] for i in pixels][::-1]
        
        reverse_img = RGBImage(pixel_reverse)
        return reverse_img

# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0 
        self.coupon = 0
        self.previous_cost = 0
        self.rotate = False 

        

    def negate(self, image):
        """
        TODO: add description

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        
        """
        # YOUR CODE GOES HERE #
        if self.coupon > 0:
            self.cost -= 1
        else:
            self.cost += 5
            
        return super().negate(image)
        

    def grayscale(self, image):
        """
        TODO: add description

        """
        # YOUR CODE GOES HERE #
        if self.coupon > 0:
            self.cost -= 1 
        else:
            self.cost += 6
            
        return super().grayscale(image)
        

    def rotate_180(self, image):
        """
        TODO: add description

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE #
        #self.rotate += 1
    
        if self.rotate == True:
            self.cost -= self.previous_cost
            self.rotate = False 
        else:
            if self.coupon > 0:
                self.coupon -= 1
                self.previous_cost = 0
            else:
                self.cost += 10
                self.previous_cost = 10
            self.rotate = True 
        return super().rotate_180(image)
            
        
        
        

    def redeem_coupon(self, amount):
        """
        TODO: add description

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        0
        """

        # YOUR CODE GOES HERE #
        if amount < 0 or amount == 0:
            raise ValueError
        
        if type(amount) != int:
            raise TypeError
        
        self.coupon += amount 

        
    

# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    TODO: add description
    """

    def __init__(self):
        """
        TODO: add description

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        TODO: add description

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
        """
        # YOUR CODE GOES HERE #
        
        chroma_pixels = chroma_image.get_pixels()
        bkg_pixels = background_image.get_pixels()
        
        if type(chroma_image) != RGBImage or type(background_image) != RGBImage:
            return TypeError
        
        if len(chroma_pixels) != len(bkg_pixels):
            return ValueError
        
        color_lst = list(color)
        
        chroma_lst = []
        
        for row in range(len(chroma_pixels)):
            temp1_lst = []
            for col in range(len(chroma_pixels[row])):
                if chroma_pixels[row][col] == color_lst:
                    temp1_lst.append(bkg_pixels[row][col])
                else:
                    temp1_lst.append(chroma_pixels[row][col])
            chroma_lst.append(temp1_lst)
        
        return RGBImage(chroma_lst)
                     
                    


    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        TODO: add description

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        sticker_pixels = sticker_image.get_pixels()
        bkg_pixels = background_image.get_pixels()
        
        if type(sticker_image) != RGBImage or type(background_image) != RGBImage:
            raise TypeError
        
        if len(sticker_pixels) >= len(bkg_pixels):
            raise ValueError
        
        if type(x_pos) != int or type(y_pos) != int:
            raise TypeError
        
        if sticker_image.size()[0] + x_pos > background_image.size()[0] or \
            sticker_image.size()[1] + y_pos > background_image.size()[1]:
            raise ValueError
               
        for row in range(len(sticker_pixels)):
            for col in range(len(sticker_pixels[row])):
                bkg_pixels[x_pos + row][y_pos + col] = sticker_pixels[row][col]

        return RGBImage(bkg_pixels)
                    
                
# Part 5: Image KNN Classifier #
def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    
    # make random training data (type: List[Tuple[RGBImage, str]])
    >>> train = []

    # create training images with low intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
    ...     for _ in range(20)
    ... )

    # create training images with high intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
    ...     for _ in range(20)
    ... )

    # initialize and fit the classifier
    >>> knn = ImageKNNClassifier(5)
    >>> knn.fit(train)

    # should be "low"
    >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    low

    # can be either "low" or "high" randomly
    >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    This will randomly be either low or high

    # should be "high"
    >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
    high
        
    """
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()

    
        
class ImageKNNClassifier:
    """
    TODO: add description
    """

    def __init__(self, n_neighbors):
        """
        TODO: add description
        """
        # YOUR CODE GOES HERE #
        self.n_neighbors = n_neighbors
        self.data = None
        

    def fit(self, data):
        """
        TODO: add description
        """
        # YOUR CODE GOES HERE #
        if self.n_neighbors > len(data):
            raise ValueError
        if self.data is not None:
            raise ValueError
        self.data = data 
        

    @staticmethod
    def distance(image1, image2):
        """
        TODO: add description
        """
        if type(image1) != RGBImage or type(image2) != RGBImage:
            raise TypeError
        if image1.size() != image2.size():
            raise ValueError
        
        #Flaten Images Pixels
        img1_pixels = image1.get_pixels()
        flat_img1_lst = [col for row in img1_pixels for col in row]
        #[[32, 45, 65], [12, 32, 89]]

        img2_pixels = image2.get_pixels()
        flat_img2_lst = [col for row in img2_pixels for col in row]
        #[[13, 24, 19], [56, 78, 92]]

        #Calculate Distance
        # dist_lst = []
        # for i in range(len(flat_img1_lst)):
        #     euclid_dist = []
        #     for j in range(len(flat_img1_lst[i])):
        #         euclid_dist.append((flat_img1_lst[i][j] - flat_img2_lst[i][j])**2)
        #     dist_lst.append(sum(euclid_dist)**(0.5))
        # return dist_lst

        dist_lst = [[(flat_img1_lst[i][j] - flat_img2_lst[i][j])**2 for j in range(len(flat_img1_lst[i]))] for i in range(len(flat_img1_lst))]
        sum_dist_lst = [sum(val)**(0.5) for val in dist_lst]
        return sum_dist_lst

    @staticmethod
    def vote(candidates):
        """
        TODO: add description
        """
        max_label = (max(set(candidates),key=candidates.count))
        return max_label
        
    def predict(self, image):
        """
        TODO: add description
        """
        euc_dists = [ImageKNNClassifier.distance(image, i[0]) for i in self.data]
       
        label_dist = [(euc_dists[i], self.data[i][1]) for i in range(len(self.data))]
        
        k = self.n_neighbors

        sorted_dist = sorted(label_dist)[:k]

        labels = [i[1] for i in sorted_dist]

        return ImageKNNClassifier.vote(labels)
            

# --------------------------------------------------------------------------- #

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)
