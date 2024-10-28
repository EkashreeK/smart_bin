import cv2
import argparse

from ultralytics import YOLO

dry_classes = (
    'water_bottle', 'pop_bottle', 'beer_bottle', 'wine_bottle', 'paper_towel', 'envelope', 'book_jacket',
    'comic_book', 'menu', 'cardboard_box', 'carton', 'plastic_bag', 'garbage_truck', 'tin_can', 'milk_can',
    'beer_can', 'pill_bottle', 'packet', 'matchstick', 'nail', 'screw', 'ballpoint_pen', 'pencil_box',
    'paper_clip', 'rubber_eraser', 'wooden_spoon', 'plate', 'bowl', 'cup', 'beer_glass', 'wine_glass',
    'coffee_mug', 'cellular_telephone', 'desktop_computer', 'notebook_computer', 'electric_fan', 'lamp',
    'television', 'CD_player', 'iPod', 'digital_clock', 'analog_clock', 'wall_clock', 'hourglass',
    'sunglasses', 'sunglass', 'binoculars', 'electric_guitar', 'acoustic_guitar', 'violin', 'cellular_telephone',
    'mouse_trap', 'projectile', 'missile', '', 'candle', 'space_shuttle', 'nail', 'paintbrush', 'syringe',
    'gasmask', 'diaper','airship', 'rubber_eraser', 'ballpoint', 'mailbag', 'toilet_tissue', 'switch',
    'knot', 'file', 'necklace', 'notebook', 'perfume','purse', 'remote_control', 'radio', 'rugby_ball',
    'scale', 'screw_driver', 'soap_dispenser', 'soup_bowl', 'spatula', 'swab', 'comic_book', 'wool',
    'wooden_spoon', 'whistle', 'whiskey_jug', 'coral_reef', 'crossword_puzzle', 'street_sign', 'hot_pot',
    'vase', 'tray', 'toaster', 'tennis_ball', 'snorkel', 'soccer_ball', 'sunscreen', 'safety_pin', 'saltshaker',
    'scoreboard', 'screwdriver', 'shopping_basket', 'parachute', 'pencil_sharpener', 'Petri_dish', 'picket_fence',
    'pinwheel', 'pirate', 'plane', 'planetarium', 'pole', 'modem', 'hook', 'lab_coat', 'measuring_cup', 'mask',
    'magnetic_compass', 'lotion', 'lighter', 'letter_opener', 'lens_cap', 'lampshade', 'ladle', 'golf_ball', 'flagpole',
    'handkerchief', 'coffeepot', 'Crock_Pot', 'croquet_ball', 'bottlecap', 'can_opener', 'carton', 'cleaver', 'bolo_tie',
    'bath_towel', 'binder', 'packet', 'conch', 'binder', 'apron', 'monitor', 'lighter', 'pay-phone', 'tree_frog', 'nematode', 'tree_frog', 'American_alligator'
)

wet_classes = (
    'banana', 'apple', 'strawberry', 'orange', 'lemon', 'pineapple', 'jackfruit', 'custard_apple', 'pomegranate',
    'fig', 'guava', 'mango', 'grapes', 'watermelon', 'cantaloupe', 'honeydew', 'cucumber', 'eggplant',
    'bell_pepper', 'head_cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti_squash', 'acorn_squash',
    'butternut_squash', 'artichoke', 'cardoon', 'mushroom', 'Granny_Smith', 'earthstar', 'stinkhorn',
    'hen-of-the-woods', 'bolete', 'corn', 'buckeye', 'organ', 'brain_coral', 'coral_fungus', 'agaric',
    'gyromitra', 'face_powder', 'ice_cream', 'pizza', 'ping_pong_ball', 'ping-pong_ball', 'dough', 'meat_loaf',
    'burrito', 'chocolate_sauce', 'eggnog', 'espresso', 'strawberry', 'butternut_squash', 'carbonara', 'potpie',
    'red_wine', 'guacamole', 'consomme', 'trifle', 'ice_lolly', 'French_loaf', 'bagel', 'pretzel', 'cheeseburger',
    'hotdog', 'mashed_potato', 'rotisserie', 'drumstick', 'snail', 'slug', 'jackfruit', 'buckeye'
)

undetermined_classes = ('buckeye', 'gasmask', 'diaper', 'Band_Aid')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOv8 Live')
    parser.add_argument(
        '--webcam-resolution',
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

args = parse_arguments()
frame_width, frame_height = args.webcam_resolution

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

cv2.namedWindow('yolov8', cv2.WINDOW_NORMAL)

# model = YOLO('./training/dataset/train_output/best.pt')
model = YOLO('yolov8m-cls.pt')

while True:
    ret, frame = cap.read()

    results = model.predict(source=frame, show=True, conf=0.7)

    if model.names[results[0].probs.top5[0]] in wet_classes or model.names[results[0].probs.top5[1]] in wet_classes or model.names[results[0].probs.top5[2]] in wet_classes or model.names[results[0].probs.top5[3]] in wet_classes or model.names[results[0].probs.top5[4]] in wet_classes:
        print('Wet')
    elif model.names[results[0].probs.top5[0]] in dry_classes or model.names[results[0].probs.top5[1]] in dry_classes:
        print('Dry')

    if not ret:
        break

    cv2.imshow('yolov8', frame)

    if cv2.waitKey(30) == 27:
        break
    

cap.release()
cv2.destroyAllWindows()