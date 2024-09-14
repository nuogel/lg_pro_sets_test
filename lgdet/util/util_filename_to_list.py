import os

 
def basename_imgpath_labpath():
    i = 5
    filepath = 'E://LG//programs//lg_pro_sets//tmp//kitti_level{}'.format(i)

    filelist = os.listdir(filepath)
    f = open('tmp/level{}_imgnames.txt'.format(i), 'w', encoding='utf-8')
    for file in filelist:
        basename = os.path.basename(file).split('.')[-2]
        imgpath = 'none'
        labpath = 'E://LG//programs//lg_pro_sets//datasets//Annotations_kitti//training//{}.xml'.format(basename[:-2])
        f.write('{};{};{}\n'.format(basename, imgpath, labpath))
    f.close()


def numbers_list():
    filepath = 'xxx'

    open(filepath, 'w')
    f = open(filepath, 'a')
    for i in range(7481):
        s = str('%06d\n' % i)
        f.write(s)
    f.close()


def just_get_basename(filepath):
    output_path = "./cache/basename.txt"
    filelist = os.listdir(filepath)
    f = open(output_path, 'w', encoding='utf-8')
    for file in filelist:
        basename = os.path.basename(file).split('.')[-2]
        f.write('{}\n'.format(basename))
    f.close()
