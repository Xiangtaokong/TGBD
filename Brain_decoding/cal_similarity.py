import numpy as np
import os
from scipy.spatial.distance import euclidean, cityblock, mahalanobis
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_regression
from concurrent.futures import ThreadPoolExecutor
import time


# voxel1,voxel2: (h, w, c)

data_list=[]

def cal_sim(subj):

    subj_ref="102311_Subj2"#102311_Subj2
    fmri_dirs='/home/notebook/data/personal/S9053103/mind_concept/Dataset/HCP/HCP_fmri/fmri_mix5s'


    fmri_path1 = os.path.join(fmri_dirs,"{}_{}.npy".format(subj_ref,filename))
    fmri_path2 = os.path.join(fmri_dirs,"{}_{}.npy".format(subj,filename))

    voxel1_flat=np.load(fmri_path1).flatten()
    voxel2_flat=np.load(fmri_path2).flatten()

   
    cosine_sim = cosine_similarity([voxel1_flat], [voxel2_flat])[0][0]

    #print("Cosine Similarity:", cosine_sim)

    data_list.append((subj,0,cosine_sim))





subjs=["100610_Subj1","102311_Subj2","104416_Subj4","105923_Subj5","108323_Subj6","109123_Subj7","111312_Subj8","111514_Subj9","114823_Subj10","115017_Subj11","115825_Subj12","116726_Subj13","118225_Subj14","125525_Subj15","126426_Subj16","126931_Subj17","128935_Subj18","130114_Subj19","130518_Subj20","131722_Subj21","132118_Subj22","134627_Subj23","134829_Subj24","135124_Subj25","137128_Subj26","140117_Subj27","144226_Subj28","145834_Subj29","146129_Subj30","146432_Subj31","146735_Subj32","146937_Subj33","148133_Subj34","150423_Subj35","155938_Subj36","156334_Subj37","157336_Subj38","158035_Subj39","158136_Subj40","159239_Subj41","162935_Subj42","164131_Subj43","164636_Subj44","165436_Subj45","167036_Subj46","167440_Subj47","169040_Subj48","169343_Subj49","169444_Subj50","169747_Subj51","171633_Subj52","172130_Subj53","173334_Subj54","175237_Subj55","176542_Subj56","177140_Subj57","177645_Subj58","177746_Subj59","178142_Subj60","178243_Subj61","178647_Subj62","180533_Subj63","181232_Subj64","182436_Subj65","182739_Subj66","185442_Subj67","186949_Subj68","187345_Subj69","191033_Subj70","191336_Subj71","191841_Subj72","192439_Subj73","192641_Subj74","193845_Subj75","195041_Subj76","196144_Subj77","197348_Subj78","198653_Subj79","199655_Subj80","200210_Subj81","200311_Subj82","200614_Subj83","201515_Subj84","203418_Subj85","204521_Subj86","205220_Subj87","209228_Subj88","212419_Subj89","214019_Subj90","214524_Subj91","221319_Subj92","233326_Subj93","239136_Subj94","246133_Subj95","249947_Subj96","251833_Subj97","257845_Subj98","263436_Subj99","283543_Subj100","318637_Subj101","320826_Subj102","330324_Subj103","346137_Subj104","352738_Subj105","360030_Subj106","365343_Subj107","380036_Subj108","381038_Subj109","385046_Subj110","389357_Subj111","393247_Subj112","395756_Subj113","397760_Subj114","401422_Subj115","406836_Subj116","412528_Subj117","429040_Subj118","436845_Subj119","463040_Subj120","467351_Subj121","525541_Subj123","541943_Subj124","547046_Subj125","550439_Subj126","562345_Subj127","572045_Subj128","573249_Subj129","581450_Subj130","601127_Subj131","617748_Subj132","627549_Subj133","638049_Subj134","644246_Subj135","654552_Subj136","671855_Subj137","680957_Subj138","690152_Subj139","706040_Subj140","724446_Subj141","725751_Subj142","732243_Subj143","745555_Subj144","751550_Subj145","757764_Subj146","765864_Subj147","770352_Subj148","771354_Subj149","782561_Subj150","783462_Subj151","789373_Subj152","814649_Subj153","818859_Subj154","825048_Subj155","826353_Subj156","833249_Subj157","859671_Subj158","861456_Subj159","871762_Subj160","872764_Subj161","878776_Subj162","878877_Subj163","898176_Subj164","899885_Subj165","901139_Subj166","901442_Subj167","905147_Subj168","910241_Subj169","926862_Subj170","927359_Subj171","942658_Subj172","943862_Subj173","951457_Subj174","958976_Subj175","966975_Subj176","971160_Subj177","995174_Subj178"]
top10_count = {}
last10_count = {}

with open('/home/notebook/data/personal/S9053103/TGBD/Datasets/HCP/all_index.txt') as f:
    filename_list=f.readlines()

with open('/home/notebook/data/personal/S9053103/brain_decoding/mindc/similarity_for_subj02_all3127.txt','w') as fw:

    count_=0
    for filename in filename_list:
        start_time = time.time()
        filename=filename.strip()
        data_list=[]

    
        with ThreadPoolExecutor(max_workers=30) as executor:
            executor.map(cal_sim, subjs)

        print("\n\nimg:{}".format(filename))
        fw.write("\n\nimg:{}".format(filename))


        sorted_by_data2_desc = sorted(data_list, key=lambda x: x[2], reverse=True)
        print("\nSorted by cosine_sim (Descending):")
        fw.write("\nSorted by cosine_sim (Descending):")
        for item in sorted_by_data2_desc[:10]:
            print(f"\nName: {item[0]}, Data1: {item[1]}, Data2: {item[2]}")
            fw.write(f"\nName: {item[0]}, Data1: {item[1]}, Data2: {item[2]}")

            name = item[0]
            if name in top10_count:
                top10_count[name] += 1
            else:
                top10_count[name] = 1

        sorted_by_data2_desc = sorted(data_list, key=lambda x: x[2])
        print("\nSorted by cosine_sim (Rising):")
        fw.write("\nSorted by cosine_sim (Rising):")
        for item in sorted_by_data2_desc[:10]:
            print(f"\nName: {item[0]}, Data1: {item[1]}, Data2: {item[2]}")
            fw.write(f"\nName: {item[0]}, Data1: {item[1]}, Data2: {item[2]}")

            name = item[0]
            if name in last10_count:
                last10_count[name] += 1
            else:
                last10_count[name] = 1
   
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Running time: {elapsed_time:.6f} s")
        count_+=1
        print('{}/3127'.format(count_))

    sorted_top10_count = sorted(top10_count.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 Count by cos (Descending):")
    fw.write("\nTop 10 Count by cos (Descending):")
    for name, count in sorted_top10_count[:20]:
        print(f"\nName: {name}, Count: {count}")
        fw.write(f"\nName: {name}, Count: {count}")

    names = [name for name, count in sorted_top10_count[:20]]
    output_string = f"\n['{', '.join(names)}']"
    print(output_string)
    fw.write(output_string)


    sorted_last_count = sorted(last10_count.items(), key=lambda x: x[1], reverse=True)

    print("\nLast 10 Count by cos (Descending):")
    fw.write("\nLast 10 Count by cos (Descending):")
    for name, count in sorted_last_count[:20]:
        print(f"\nName: {name}, Count: {count}")
        fw.write(f"\nName: {name}, Count: {count}")

    names = [name for name, count in sorted_last_count[:20]]
    output_string = f"\n['{', '.join(names)}']"
    print(output_string)
    fw.write(output_string)

