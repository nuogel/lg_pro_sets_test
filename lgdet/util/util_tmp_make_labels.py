import glob
import xmltodict


if __name__ == '__main__':
    lab_path = '/media/dell/data/比赛/VocXml/labels'
    xmls = glob.glob(lab_path+'/*.xml')
    names = []
    for anno_path in xmls:
        datas = xmltodict.parse(open(anno_path, mode='r').read())
        for data in datas['annotation']['object']:
            name = data['name']
            names.append(name)

    names = list(set(names))
    names.sort()
    print(len(names))

    for name in names:
        print(name)
