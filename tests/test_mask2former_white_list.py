from ai_utils.Mask2FormerInference import Mask2FormerInference
import cv2

mask2former = Mask2FormerInference(model_weights="/home/frollo/ros_workspaces/ros2/nav_ws/src/artifacts_mapping/artifacts_mapping_lib/weights/model_final_e0c58e.pkl", 
config_file="/home/frollo/ros_workspaces/ros2/nav_ws/src/artifacts_mapping/artifacts_mapping_lib/net_config/maskformer2_swin_large_IN21k_384_bs16_160k.yaml"
)

# !!IMPORTANT!! We are doing tests with an instance of mask2former class. 
# Test order is important because member variables are not reinitialized during tests

# ADD TESTS
def test_add_single_class_in_empty_set():
    mask2former.add_class("cacca")
    assert mask2former.classes_white_list == {"cacca"}

def test_add_single_class_in_set_with_same_class():
    mask2former.add_class("cacca")
    assert mask2former.classes_white_list == {"cacca"}

def test_add_list():
    mask2former.add_class(["prova", "lista"])
    assert mask2former.classes_white_list == {"cacca", "prova", "lista"}

def test_add_set():
    mask2former.add_class({"try", "set"})
    assert mask2former.classes_white_list == {"cacca", "prova", "lista", "try", "set"}

def test_add_list_in_set_with_same_class():
    mask2former.add_class("cacca")
    assert mask2former.classes_white_list == {"cacca", "prova", "lista", "try", "set"}

def test_set_in_set_with_same_class():
    mask2former.add_class("cacca")
    assert mask2former.classes_white_list == {"cacca", "prova", "lista", "try", "set"}

# REMOVE TESTS
def test_remove_class_from_set_where_present():
    mask2former.remove_class("cacca")
    assert mask2former.classes_white_list == {"prova", "lista", "try", "set"}

def test_remove_list_from_set_where_present():
    mask2former.remove_class(["prova", "lista"])
    assert mask2former.classes_white_list == {"try", "set"}

def test_remove_set_from_set_where_present():
    mask2former.remove_class({"try", "set"})
    assert mask2former.classes_white_list == set()

def test_remove_class_from_set_where_not_present():
    mask2former.remove_class("cacca")
    assert mask2former.classes_white_list == set()

def test_remove_list_from_set_where_not_present():
    mask2former.remove_class(["prova", "lista"])
    assert mask2former.classes_white_list == set()

def test_remove_set_from_set_where_not_present():
    mask2former.remove_class({"try", "set"})
    assert mask2former.classes_white_list == set()

# CHECK IF SET FUNCTION ARE WORKING
def test_set_is_empty():
    assert not mask2former.classes_white_list

def test_class_not_in_white_list():
    assert not ("cacca" in mask2former.classes_white_list)

def test_set_is_not_empty():
    mask2former.add_class("cacca")
    assert mask2former.classes_white_list

def test_class_not_in_white_list():
    assert "cacca" in mask2former.classes_white_list


def test_clear_white_list():
    mask2former.add_class(["prova", "lista"])
    mask2former.add_class({"try", "set"})
    mask2former.empty_white_list()
    assert mask2former.classes_white_list == set()

# TEST IMAGE PREDICTION FILTERING
image = cv2.imread("tests/test_image.png")

def test_prediction_with_no_filtering():
    pred = mask2former.img_inference(image)
    assert list(pred.keys()) == ['box', 'chair', 'ceiling', 'wall', 'bottle', 'light', 'cabinet', 'person', 'floor']

def test_prediction_with_filtering_if_class_not_present():
    mask2former.add_class("cacca")
    pred = mask2former.img_inference(image)
    assert list(pred.keys()) == list()

def test_prediction_with_filtering_if_class_is_present():
    mask2former.add_class("person")
    pred = mask2former.img_inference(image)
    assert list(pred.keys()) == ['person']

def test_prediction_with_filtering_if_class_is_removed():
    mask2former.remove_class("person")
    pred = mask2former.img_inference(image)
    assert list(pred.keys()) == list()

def test_prediction_with_filtering_if_multiple_class_added():
    import collections
    mask2former.add_class(["person", "chair"])
    pred = mask2former.img_inference(image)
    assert collections.Counter(list(pred.keys())) == collections.Counter(['person', 'chair'])

def test_prediction_with_filtering_if_class_removed_but_white_list_not_empty():
    mask2former.remove_class("person")
    pred = mask2former.img_inference(image)
    assert list(pred.keys()) == ['chair']

def test_prediction_with_filtering_when_multiple_classes_are_removed_and_white_list_empty():
    mask2former.remove_class(["person", "chair", "cacca"])
    print(mask2former.classes_white_list)
    pred = mask2former.img_inference(image)
    assert list(pred.keys()) == ['box', 'chair', 'ceiling', 'wall', 'bottle', 'light', 'cabinet', 'person', 'floor']