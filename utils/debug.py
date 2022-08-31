import os
import shutil
import time
import uuid

import cv2
import torch
import numpy as np
from PIL import Image
from xml.dom.minidom import Document

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

LICENSE_HELMET_TO_CHAR = [
    {"field": "name", "oval": "0", "nval": "no_helmet"},
    {"field": "name", "oval": "1", "nval": "have_helmet"},
    {"field": "name", "oval": "2", "nval": "license"},
    {"field": "name", "oval": "3", "nval": "rider"},
]

LICENSE_HELMET_TO_NUM = [
    {"field": "name", "oval": "no_helmet", "nval": "0"},
    {"field": "name", "oval": "have_helmet", "nval": "1"},
    {"field": "name", "oval": "license", "nval": "2"},
    {"field": "name", "oval": "rider", "nval": "3"},
]


def cxy2xyxy(size, bbox):
    h, w = size
    cx, cy, cw, ch = list(map(float, bbox))

    x1 = (cx * w) - ((cw * w) / 2)
    y1 = (cy * h) - ((ch * h) / 2)
    x2 = x1 + (cw * w)
    y2 = y1 + (ch * h)

    return list(map(int, [x1, y1, x2, y2]))


class GTBox:

    @staticmethod
    def draw_rectangle(src, locations):
        if isinstance(src, Image.Image):
            src = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)

        for location in locations:
            x1, y1, x2, y2 = location
            src = cv2.rectangle(src, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('show', src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def gt_from_voc(image_root, label_root, save_root=None):
        for _, _, image_list in os.walk(image_root):
            for idx, item in enumerate(image_list):
                image_name, _, suffix = item.rpartition(".")

                image_path = os.path.join(image_root, item)
                label_path = os.path.join(label_root, f'{image_name}.xml')
                xml_root = ET.ElementTree(file=label_path).getroot()

                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                for gt_obj in xml_root.iter('object'):
                    gt_box = gt_obj.find('bndbox')

                    x1 = int(gt_box.find('xmin').text)
                    y1 = int(gt_box.find('ymin').text)
                    x2 = int(gt_box.find('xmax').text)
                    y2 = int(gt_box.find('ymax').text)

                    if str(gt_obj.find('name').text) == "0":
                        color = (255, 0, 0)
                    elif str(gt_obj.find('name').text) == "2":
                        color = (0, 255, 0)
                    elif str(gt_obj.find('name').text) == "3":
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 0)

                    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                    image = cv2.putText(
                        image, str(gt_obj.find('name').text), (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color, 1
                    )

                if save_root is not None:
                    save_path = os.path.join(save_root, f'{image_name}.{suffix}')
                    if cv2.imwrite(save_path, image):
                        print(f'【{idx}】write {image_name} successfully')
                else:
                    cv2.imshow(f"{idx}-{image_name}", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    @staticmethod
    def gt_form_coco(image_root, label_root, save_root=None):
        for _, _, image_list in os.walk(image_root):
            for idx, item in enumerate(image_list):
                image_name, _, suffix = item.rpartition(".")

                image_path = os.path.join(image_root, item)
                label_path = os.path.join(label_root, f'{image_name}.txt')

                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
                img_shape = image.shape[:2]  # h, w

                with open(label_path, encoding='utf-8') as f:
                    lines = f.readlines()

                text = ""
                for _, line in enumerate(lines):
                    box_info = line.strip("\n").split(" ")
                    cls = box_info[0]
                    x1, y1, x2, y2 = cxy2xyxy(img_shape, list(map(float, box_info[1:])))

                    text += " " + str(cls)
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                image = cv2.putText(image, f"class={text}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if save_root is not None:
                    save_path = os.path.join(save_root, f'gt-{image_name}.{suffix}')
                    if cv2.imwrite(save_path, image):
                        print(f'【{idx}】write {image_name} successfully')
                else:
                    cv2.imshow(f"{idx}-{image_name}", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


class ImageUtils:
    @staticmethod
    def img2cv(src):
        if isinstance(src, Image.Image):
            src = cv2.cvtColor(np.array(src), cv2.COLOR_RGB2BGR)
        return src

    @staticmethod
    def cv2img(src):
        if isinstance(src, np.ndarray):
            src = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        return src

    @staticmethod
    def show_cv_img(src, title="origin"):
        data = ImageUtils.img2cv(src)
        cv2.imshow(title, data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_img(src, title="origin"):
        data = ImageUtils.cv2img(src)
        data.show(title)

    @staticmethod
    def show_img_groups(groups):
        srcs = [s[0] for s in groups]
        datas = list(map(ImageUtils.img2cv, srcs))
        datas = np.hstack(datas)

        title = ' '.join([s[1] for s in groups])
        cv2.imshow(datas, title)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def resize(src, scale, mode='cv', save=False, dst=None):
        if os.path.isdir(src):
            root_path = src
            src = [os.path.join(root_path, item) for item in os.listdir(src)]
        else:
            root_path = os.path.dirname(src)
            src = [src]

        for item in src:
            filename, _, suffix = os.path.basename(item).rpartition(".")

            if suffix not in ["jpg", "png", "JPG", "jpeg"]:
                continue

            data = cv2.imread(item)
            if data is None:
                print(f'【error】：read file {filename} error')

            rd = cv2.resize(data, scale)
            if save:
                if dst is None:
                    cv2.imwrite(os.path.join(root_path, f"{filename}_resized.{suffix}"), rd)
                else:
                    cv2.imwrite(os.path.join(dst, f"{filename}_resized.{suffix}"), rd)
                print(f"resized {filename} successfully")
            else:
                return rd


class DatasetUtils:
    @staticmethod
    def voc2coco(xml_root_path, label_root_path):
        for _, dirs, xml_list in os.walk(xml_root_path):
            for i, item in enumerate(xml_list):
                xml_name, _, _ = item.rpartition(".")

                xml_path = os.path.join(xml_root_path, item)
                txt_path = os.path.join(label_root_path, f'{xml_name}.txt')

                txt = open(txt_path, 'w')
                xml_root = ET.parse(xml_path).getroot()
                size = xml_root.find('size')
                w_image = float(size.find('width').text)
                h_image = float(size.find('height').text)

                for gt_obj in xml_root.iter('object'):
                    obj_name = gt_obj.find('name').text
                    gt_box = gt_obj.find('bndbox')

                    x_min = float(gt_box.find('xmin').text)
                    x_max = float(gt_box.find('xmax').text)
                    y_min = float(gt_box.find('ymin').text)
                    y_max = float(gt_box.find('ymax').text)

                    x_center = ((x_min + x_max) / 2) / w_image
                    y_center = ((y_min + y_max) / 2) / h_image
                    w = (x_max - x_min) / w_image
                    h = (y_max - y_min) / h_image

                    txt.write(f"{obj_name} {x_center} {y_center} {w} {h}\n")

                txt.close()
                print(f"【{i}】parse {xml_path} successfully")

    @staticmethod
    def coco2voc(root_path):
        img_type = "jpg"

        txt_root = os.path.join(root_path, "labels")
        image_root = os.path.join(root_path, "images")
        xml_root = os.path.join(root_path, "xmls")

        for i, file in enumerate(os.listdir(txt_root)):
            if not file.endswith(".txt"):
                print(f"【error】: file {file} is not txt")
                break
            else:
                txt_base_name, _, _ = file.partition(".txt")
                txt = open(os.path.join(txt_root, file))

            try:
                h, w, c = cv2.imread(os.path.join(image_root, f'{txt_base_name}.{img_type}')).shape
            except Exception as _:
                raise f"【error】: img {txt_base_name} is not {img_type}"

            xml_builder = Document()
            annotation = xml_builder.createElement("annotation")  # 创建annotation标签
            xml_builder.appendChild(annotation)

            filename = xml_builder.createElement("filename")  # filename标签
            filename.appendChild(xml_builder.createTextNode(f'{txt_base_name}.{img_type}'))
            annotation.appendChild(filename)

            size = xml_builder.createElement("size")  # size标签
            width = xml_builder.createElement("width")  # size子标签width
            width.appendChild(xml_builder.createTextNode(str(w)))
            size.appendChild(width)

            height = xml_builder.createElement("height")  # size子标签height
            height.appendChild(xml_builder.createTextNode(str(h)))
            size.appendChild(height)

            depth = xml_builder.createElement("depth")  # size子标签depth
            depth.appendChild(xml_builder.createTextNode(str(c)))
            size.appendChild(depth)
            annotation.appendChild(size)

            boxes = txt.readlines()
            for box in boxes:
                oneline = box.strip().split(" ")
                obj = xml_builder.createElement("object")

                cls = xml_builder.createElement("name")
                cls.appendChild(xml_builder.createTextNode(oneline[0]))
                obj.appendChild(cls)

                bndbox = xml_builder.createElement("bndbox")
                xmin_val, ymin_val, xmax_val, ymax_val = cxy2xyxy((w, h), oneline[1:])
                xmin = xml_builder.createElement("xmin")
                xmin.appendChild(xml_builder.createTextNode(str(xmin_val)))
                bndbox.appendChild(xmin)

                ymin = xml_builder.createElement("ymin")
                ymin.appendChild(xml_builder.createTextNode(str(ymin_val)))
                bndbox.appendChild(ymin)

                xmax = xml_builder.createElement("xmax")
                xmax.appendChild(xml_builder.createTextNode(str(xmax_val)))
                bndbox.appendChild(xmax)

                ymax = xml_builder.createElement("ymax")
                ymax.appendChild(xml_builder.createTextNode(str(ymax_val)))
                bndbox.appendChild(ymax)
                obj.appendChild(bndbox)

                annotation.appendChild(obj)

            f = open(os.path.join(xml_root, f"{txt_base_name}.xml"), 'w')
            xml_builder.writexml(f, indent='\t', newl='\n', addindent="\t", encoding='utf-8')
            f.close()

            print(f"【{i}】 {file} convert successfully")

    @staticmethod
    def update_xml_attribute(dir_root, updates):
        old_obj_info = DatasetUtils.show_label_info(dir_root)

        for _, _, xml_list in os.walk(dir_root):
            for idx, item in enumerate(xml_list):
                if not item.endswith(".xml"):
                    continue
                xml_path = os.path.join(dir_root, item)

                doc = ET.parse(xml_path)
                xml_root = doc.getroot()

                for u in updates:
                    field = u.get("field")

                    if field == "filename":
                        # att = xml_root.find(field)
                        # att.text = field.get("nval")
                        pass

                    elif field == "path":
                        pass

                    elif field == "name":
                        for gt_obj in xml_root.iter('object'):
                            att = gt_obj.find('name')
                            if att.text == u.get("oval"):
                                att.text = u.get("nval")
                    else:
                        print(f"【error - {xml_path}】 不支持修改的标签")

                doc.write(xml_path)
                print(f"【{idx}】update {xml_path} successfully")

        new_obj_info = DatasetUtils.show_label_info(dir_root)

        print(f'old_obj_info: {old_obj_info}; new_obj_info: {new_obj_info}')

    @staticmethod
    def delete_xml_obj(dir_root, field):
        for _, _, xml_list in os.walk(dir_root):
            for idx, item in enumerate(xml_list):
                if not item.endswith(".xml"):
                    continue
                xml_path = os.path.join(dir_root, item)

                doc = ET.parse(xml_path)
                xml_root = doc.getroot()
                for gt_obj in xml_root.iter('object'):
                    if gt_obj.find('name').text == field:
                        xml_root.remove(gt_obj)

                doc.write(xml_path)
                print(f"【{idx}】update {xml_path} successfully")

    @staticmethod
    def clean_label(image_root, label_root):
        """删除不存在图片的 label"""
        for _, _, label_list in os.walk(label_root):
            for idx, item in enumerate(label_list):
                label_name, _, suffix = item.rpartition(".")

                image_path_jpg = os.path.join(image_root, f'{label_name}.jpg')
                image_path_png = os.path.join(image_root, f'{label_name}.png')

                if not (os.path.exists(image_path_jpg) or os.path.exists(image_path_png)):
                    os.remove(os.path.join(label_root, item))
                    print(f'clean {item} successfully')

    @staticmethod
    def clean_images(image_root, label_root):
        """删除不存在 label 的 image"""
        for _, _, image_list in os.walk(image_root):
            for idx, item in enumerate(image_list):
                image_name, _, suffix = item.rpartition(".")

                label_path_txt = os.path.join(label_root, f'{image_name}.txt')
                label_path_xml = os.path.join(label_root, f'{image_name}.xml')

                if not (os.path.exists(label_path_txt) or os.path.exists(label_path_xml)):
                    os.remove(os.path.join(image_root, item))
                    print(f'clean {item} successfully')

    @staticmethod
    def clean(image_root, label_root, compare_root):
        for _, _, image_list in os.walk(image_root):
            for idx, item in enumerate(image_list):
                image_name, _, suffix = item.rpartition(".")

                if not os.path.exists(os.path.join(compare_root, item)):
                    image_path = os.path.join(image_root, item)
                    label_path = os.path.join(label_root, image_name)

                    os.remove(image_path)
                    if os.path.exists(f'{label_path}.txt'):
                        os.remove(f'{label_path}.txt')
                    else:
                        os.remove(f'{label_path}.xml')

                    print(f'clean {item} successfully')

    @staticmethod
    def rename(image_root, label_root):
        for _, _, image_list in os.walk(image_root):
            for idx, item in enumerate(image_list):
                image_name, _, suffix = item.rpartition(".")
                new_name = f'{str(uuid.uuid4())[:10]}-{int(time.time())}'
                # new_name = f'{str(uuid.uuid4())}'

                image_path = os.path.join(image_root, item)
                label_path = os.path.join(label_root, f'{image_name}')

                os.rename(image_path, os.path.join(image_root, f'{new_name}.{suffix}'))
                if os.path.exists(f'{label_path}.txt'):
                    os.rename(f'{label_path}.txt', os.path.join(label_root, f'{new_name}.txt'))
                else:
                    os.rename(f'{label_path}.xml', os.path.join(label_root, f'{new_name}.xml'))

                print(f'【{idx}】 rename {image_name} successfully')

    @staticmethod
    def show_label_info(label_root):
        obj_info = {}
        for i, item in enumerate(os.listdir(label_root)):
            if not item.endswith(".xml"):
                continue

            xml_root = ET.parse(os.path.join(label_root, item)).getroot()
            for gt_obj in xml_root.iter('object'):
                obj_name = gt_obj.find('name').text

                if obj_name == "delete":
                    print(f"remove {item}")
                    os.remove(os.path.join(label_root, item))
                    break

                if obj_info.get(obj_name) is None:
                    obj_info[obj_name] = 1
                else:
                    obj_info[obj_name] += 1

        return obj_info

    @staticmethod
    def remove_argument_images(image_root):
        prefixes = []
        for idx, item in enumerate(os.listdir(image_root)):
            prefix, _, suffix = item.partition(".")
            if not (item.endswith(".jpg") or item.endswith(".png")):
                continue

            if prefix not in prefixes:
                prefixes.append(prefix)
            else:
                os.remove(os.path.join(image_root, item))
                print(f'remove {item} successfully')

    @staticmethod
    def rename1(image_root, label_root):
        for _, _, image_list in os.walk(image_root):
            for idx, item in enumerate(image_list):
                item.replace("gt-", "")
                new_name = item.replace("gt-", "")

                os.rename(
                    os.path.join(image_root, item), os.path.join(image_root, new_name)
                )
                print(f'【{idx}】 rename {item} ==> {new_name} successfully')


class ModelStructure:

    @staticmethod
    def show_model_shape_v5(layers):
        from models import common

        outputs = []
        ipt = torch.Tensor(1, 3, 640, 640)
        print(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}  {'out.shape': <30}")

        for idx, layer in enumerate(layers):
            old_ipt = ipt
            if isinstance(layer, common.Concat):
                x = outputs[layer.f[-1]]
                ipt = torch.cat((x, ipt), dim=1)
                outputs.append(ipt)
            elif layer.type == 'Detect':
                continue
            else:
                ipt = layer(ipt)
                outputs.append(ipt)

            print(f'{layer.i:>3}{str(layer.f):>18}{layer.n:>3}{layer.np:10.0f}  {layer.type:<40}{str(layer.args):<30} '
                  f'{",".join(list(map(str, list(old_ipt.shape))))} => {",".join(list(map(str, list(ipt.shape))))}')
            # print(f"【{idx}】", layer, "out.shape => ", ipt.shape)

    # @staticmethod
    # def show_model_shape_v7(layers):
    #     outputs = []
    #     ipt = torch.Tensor(1, 3, 640, 640)
    #     for idx, layer in enumerate(layers):
    #         if isinstance(
    #                 layer,
    #                 (Conv, MP, SPPCSPC, torch.nn.modules.upsampling.Upsample, RepConv, PatchEmbed,
    #                  SwinTransformer2Block, CoordAtt)
    #         ):
    #             c_ipt = ipt
    #             if len(outputs) == 0:
    #                 ipt = layer(ipt)
    #             else:
    #                 ipt = layer(outputs[layer.f])
    #
    #         elif isinstance(layer, Concat):
    #             c_ipt = ipt
    #             ipt = torch.cat([outputs[i] for i in layer.f], dim=1)
    #
    #         elif isinstance(layer, IDetect):
    #             pass
    #
    #         else:
    #             print(f"【error】： {idx, c_ipt.shape, ipt.shape}")
    #             return
    #
    #         outputs.append(ipt)
    #         logger.info(
    #             '%3s%18s%3s%10.0f  %-40s%-30s%5s' %
    #             (
    #                 layer.i, layer.f, layer.n, layer.np, layer.type, layer.args,
    #                 f"{','.join([str(i) for i in c_ipt.shape])} ==> {','.join([str(i) for i in ipt.shape])}"
    #             )
    #         )


if __name__ == '__main__':
    print("hello world")

    print(DatasetUtils.show_label_info(r'E:\datas\py\datasets\helmet_license\valid\xmls'))

    # DatasetUtils.update_xml_attribute(
    #     r'E:\datas\py\datasets\helmet_license\train\xmls',
    #     LICENSE_HELMET_TO_NUM
    # )

    # GTBox.gt_from_voc(
    #     r'E:\datas\py\datasets\helmet_license\train\images',
    #     r'E:\datas\py\datasets\helmet_license\train\xmls',
    #     r'E:\datas\py\datasets\helmet_license\train\check',
    # )

    # GTBox.gt_form_coco(
    #     r'E:\datas\py\datasets\helmet_license\valid\images',
    #     r'E:\datas\py\datasets\helmet_license\valid\labels',
    #     r'E:\datas\py\datasets\helmet_license\valid\check',
    # )

    # DatasetUtils.delete_xml_obj(
    #     r'E:\datas\py\plate\01\train\xmls',
    #     "Rider",
    # )
    # DatasetUtils.show_label_info(r'E:\datas\py\plate\01\train\xmls')

    # DatasetUtils.voc2coco(
    #     r'E:\datas\py\datasets\helmet_license\valid\xmls',
    #     r'E:\datas\py\datasets\helmet_license\valid\labels'
    # )

    # DatasetUtils.clean_label(
    #     r'E:\datas\py\datasets\helmet_license\valid\images',
    #     r'E:\datas\py\datasets\helmet_license\valid\xmls'
    # )
    # DatasetUtils.clean_images(
    #     r'E:\datas\py\datasets\helmet_license\valid\images',
    #     r'E:\datas\py\datasets\helmet_license\valid\xmls'
    # )
    # DatasetUtils.remove_argument_images(r'E:\datas\py\plate\04\train\images')
    # DatasetUtils.clean(
    #     r'E:\datas\py\datasets\helmet_license\train\images',
    #     r'E:\datas\py\datasets\helmet_license\train\xmls',
    #     r'E:\datas\py\datasets\helmet_license\train\check',
    # )

    # DatasetUtils.rename(
    #     r'E:\datas\py\datasets\helmet_license\train\images',
    #     r'E:\datas\py\datasets\helmet_license\train\xmls'
    # )

    # DatasetUtils.rename1(
    #     r'E:\datas\py\datasets\helmet_license\train\check',
    #     r'E:\datas\py\datasets\helmet_license_all_v1\helmet_license\plate\train\xmls'
    # )

    # ImageUtils.resize(
    #     r"E:\datas\py\datasets\helmet_license\train\new", scale=(640, 640), mode='cv',
    #     save=True,
    # )
