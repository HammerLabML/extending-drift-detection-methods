#!/usr/bin/env python3
import pickle
import random
import sys
from copy import deepcopy
from functools import cmp_to_key
from multiprocessing import Pool
import numpy as np
from time import time
import matplotlib.pyplot as plt

from datasets import rialto_dataset, mnist_dataset, covtype_dataset, trivial_dataset, music_dataset
from d3 import D3
from MMDDDM import MMDDDM
from HDDDM import HDDDM
from SpectralDDM import SpectralDDM

# seeds are random integers from randint(0, 2147483648)
seeds_for_runs = [1135693273, 2066865442, 228000554, 1258519452, 1801586799, 345259330, 1181649071, 861477858, 1385974904, 827529294, 1963154439, 812656480, 2035051016, 1351874643, 298608514, 1595758943, 2033129634, 596658593, 1218768463, 1762730971, 579976431, 1639315236, 302709520, 919588676, 1668457611, 1190987304, 1247338838, 1519691065, 833430193, 426301061, 145651725, 1083322366, 624448727, 412036737, 1394928859, 1760843418, 953494301, 1998146258, 1473304678, 1715451564, 535628446, 392257242, 396467037, 1245281769, 408881311, 1202743020, 740624242, 2037706439, 483418314, 1818624504, 91457987, 572192666, 431498098, 1285836832, 181324312, 1528218991, 1691740479, 1619891549, 304252683, 1248323471, 362855316, 290997060, 966470762, 1473786556, 321761313, 2047932120, 271130893, 659748718, 304171980, 1872741026, 985851156, 1678194130, 343032664, 776655626, 1156690640, 38776913, 861875582, 951471516, 758913631, 2067217896, 180330285, 1572698061, 1541753822, 602928736, 301868528, 1113325032, 1775214586, 862493472, 799481370, 235717215, 1875966919, 204611447, 548101708, 171489109, 687733067, 1507213441, 1931894046, 87936577, 1079486197, 173791656, 1432483093, 938137965, 249526076, 1892660294, 668771840, 1433033976, 289221679, 1031306403, 1793973079, 76563434, 382530400, 1245378196, 2126536074, 800481373, 76623070, 1901458154, 1900070063, 654146787, 574107167, 521123265, 1418959861, 318949936, 149343809, 1823719615, 970084571, 646232018, 594216028, 1049973985, 1054188555, 445354855, 1889920667, 1322890423, 1154470569, 1285368496, 352268931, 656820676, 606764011, 1275003308, 2067440593, 1897612269, 731901958, 283821477, 1395241338, 982279218, 797176054, 2016959642, 1952420071, 1353225563, 2061567307, 816931848, 585790319, 162151837, 1498609843, 46790503, 1033825766, 1926187662, 1969566919, 386795735, 82061935, 188927986, 1476392414, 571799380, 1034942739, 2132605265, 1829160135, 413098184, 1406642491, 137647033, 200889675, 656804038, 1136382684, 757516833, 1248008169, 1249364833, 1695436078, 1826220651, 629462996, 1774867532, 366205515, 2093887790, 1516662106, 233289234, 1037763567, 1277975910, 1168150738, 1790440833, 1855228212, 2038155270, 466823705, 1790753444, 16888597, 1273770028, 1099652147, 1127091349, 926368680, 1824589408, 744588187, 2027431722, 868066963, 1832299072, 716649808, 73939015, 511362153, 817653029, 1236864217, 194783853, 2058053371, 660990242, 171556379, 82703132, 947263627, 547849501, 2060849316, 1998735588, 1857313453, 778651716, 41635175, 227492907, 1349947754, 833506241, 2060064702, 406082054, 159738951, 2016521535, 1710175595, 126992601, 1412304930, 1677624200, 158442573, 43591215, 850447906, 2100560640, 962333284, 1360974037, 451147403, 1154474381, 411221827, 2001497152, 1249668922, 1065053249, 520420385, 1384708527, 604442692, 1100180369, 257804444, 236383716, 1692219434, 1134992768, 1999444855, 858734235, 1813871282, 1696267676, 926387521, 1465882562, 255673871, 717057766, 2115930228, 415197713, 1851621312, 482111395, 890949787, 1130290479, 63796489, 600219969, 1516453401, 1536754167, 716405602, 947531515, 2126882648, 479514984, 1806305491, 227935859, 418224712, 1984315762, 1094697828, 195750791, 318991886, 286279418, 1675092109, 1295805643, 1075457388, 1283861717, 861515704, 1630163605, 2038352641, 245396676, 545351458, 1955026551, 1688047763, 1916717613, 1728115567, 453653324, 694601956, 479248320, 1640629903, 1637245371, 1812917219, 2059643738, 995967160, 1614732656, 1604470515, 1049620717, 1060852530, 9077512, 1790751136, 1587131571, 1548924808, 935948224, 127067985, 1668263127, 926626324, 331547247, 1405032901, 1637133273, 596298960, 812243423, 926088604, 1959696990, 1085371687, 716819963, 115906012, 234408117, 966469888, 659987683, 271299399, 2004129483, 135181028, 339300181, 691400509, 820396565, 760970505, 209706163, 1561197452, 550713361, 801768867, 337037846, 834521899, 1494649718, 696366051, 1818716614, 1811931323, 1859933332, 767229564, 1030776653, 211677389, 1459299840, 1821559531, 1977733123, 1889832779, 378636458, 577945666, 494354966, 991849652, 1899892125, 436415108, 604620637, 510564868, 1531169828, 773255157, 275816399, 1387729419, 1709896515, 1153858392, 893288772, 1403572348, 1766894291, 435855950, 291837366, 527395036, 537934138, 2001296593, 1264853278, 2011809004, 803182741, 1188103163, 2056245850, 787816820, 1121753743, 1630689146, 62021524, 512953835, 706582562, 763962253, 1324021902, 1863647522, 2122054871, 1388012241, 638293022, 816370742, 1602516660, 1699005740, 723615890, 1699827882, 1349023442, 828072009, 512497920, 2018016536, 494973827, 286468046, 403857609, 2086353986, 1412361662, 1204715443, 832725142, 2091181908, 1809596712, 967664095, 1270180863, 1555601423, 1946765139, 1725469695, 843628833, 1398718122, 617410596, 158027915, 627626918, 1160140084, 39961520, 1636665829, 614138036, 1216485290, 933866238, 967209892, 64608081, 1114622811, 1536750872, 1857297964, 1111262892, 906727941, 1400669336, 1676841675, 2118481463, 24134303, 538288077, 38497312, 638389702, 1226201514, 1813928435, 1880984712, 1968565285, 1160065627, 1873716361, 1683702603, 268630470, 1097421281, 2038460411, 49542427, 462778047, 1238200401, 1658674387, 2045336418, 2003493365, 140657017, 1344574666, 582202762, 1963997857, 1720446870, 1670968813, 1889174613, 1425642252, 563574005, 1538807770, 1206860122, 175867137, 1018858795, 1822848038, 1257483986, 167627659, 850937554, 1608325974, 2100097916, 1516551828, 2031860951, 1012295990, 98870122, 185559572, 1517861199, 1496471426, 170565222, 1261404800, 495968510, 117233053, 389802315, 1207248073, 1994197579, 1357482651, 1050599310, 846663532, 1234121727, 1724609354, 228991497, 1857955200, 1187472431, 1842258488, 85527239, 287546639, 596773571, 914016058, 539360370, 1019805736, 1114705183, 171308819, 753057639, 1272218509, 2110451663, 1129244631, 1859922722, 2052444133, 690735301, 1369127565, 811000802, 475456406, 904363260, 856350974, 1410254254, 1663459739, 1051740992, 104341216, 276231078, 981856905, 1557018640, 908414354, 1809196553, 1904934768, 745291253, 1027025804, 470934540, 149195512, 564806228, 1809156416, 1575019763, 1034013674, 1907701342, 1336977333, 674629026, 1458299807, 1587094006, 382571745, 54482292, 2078155970, 1835998656, 1315182169, 1666647450, 505605831, 2049430810, 805037828, 624074934, 1066181028, 384066168, 1453295696, 1947819623, 618110806, 642050965, 1320638205, 1184657197, 2070451336, 1375193117, 680751887, 1556852877, 669639703, 777405834, 721092136, 1344078025, 235269097, 753248623, 952551293, 1700520229, 584010922, 1081696337, 1265145107, 1433281584, 1555440047, 1337523089, 466307345, 736596010, 2112288669, 850026594, 1429584542, 1666842562, 528774747, 575234992, 2124375175, 1194456822, 1080294872, 1422305792, 2009727377, 916354594, 1814910182, 394730221, 1722866991, 1333289108, 1164804266, 326930027, 922765895, 941945361, 2079201304, 1022008981, 46181494, 2051606741, 1798444193, 2000164654, 2016367986, 1192003589, 633648789, 938421868, 1795668103, 736893680, 1741976363, 261277114, 1766574746, 101780413, 1870691937, 1035417869, 582291315, 1100583251, 1530583216, 1965948626, 741360178, 1629127944, 1953758548, 1603019197, 1304531010, 1269871334, 1486712766, 327927508, 44644999, 1750872017, 1445477, 761997823, 1302874439, 499906372, 2032836184, 818427587, 632983802, 41217242, 2086248229, 1096103679, 946669015, 1077752635, 164563741, 1017287427, 315312931, 1396212786, 1115979737, 1809398290, 1196784512, 1967407460, 2034274830, 647186560, 377571710, 518945672, 1784937270, 55724763, 334954689, 1116903429, 789306702, 2127510491, 908307443, 1797111909, 2091291384, 376206393, 789628197, 651218559, 683155663, 1691845393, 783568673, 97060800, 204584436, 801724299, 1490795693, 111935131, 1940333387, 937798895, 28229905, 784238016, 1081777768, 982149377, 1949357056, 458773656, 1632735426, 240968426, 1781005385, 86512035, 55109178, 868463290, 74993940, 333326019, 148189961, 1763947662, 1864356774, 1812496701, 1095685153, 1913967869, 799326273, 762353132, 2130059595, 1263857574, 2046856757, 2096706164, 918728843, 2016729638, 2105102444, 1899856264, 745046218, 2116925505, 981289166, 1850727239, 1069735591, 1330371111, 1268602557, 1114170656, 196776386, 285236551, 1494604948, 1153914543, 452593366, 714626448, 746473969, 701261708, 1653570299, 20831712, 227782707, 696386096, 573883748, 486996649, 1613933176, 1527723299, 378464388, 851013306, 1657076742, 239540596, 175459476, 1518297324, 276974315, 1088120987, 1968257843, 480052068, 212432788, 418306124, 1942160687, 99014525, 1325390379, 545978799, 651213186, 957979427, 1817125724, 1311464031, 1979685719, 23476516, 1121157417, 1799055953, 1507631119, 454401117, 1078916462, 1555481432, 829462212, 1653424752, 1046827888, 681646131, 29116240, 237106008, 1062588432, 612050497, 905343590, 2766062, 1529980867, 622361814, 15009504, 2037803655, 1677934247, 238272927, 1492811255, 1507542551, 925913996, 1490065419, 60329437, 975280521, 930437492, 2031203349, 1153101695, 1615354835, 1898443257, 1512535678, 878431251, 484249047, 528213481, 1495758594, 1141228392, 1487009968, 55439617, 919086741, 1393880584, 22116411, 1053695428, 1063098985, 1509223874, 2069539487, 782048449, 2055568107, 577037930, 568456542, 1203074229, 414490826, 924464246, 848534198, 459189343, 627959978, 1082978347, 1507077012, 1123973031, 956467920, 1412093418, 1579526053, 388112553, 1350875337, 35862202, 6875110, 486944583, 442878725, 950337246, 480030802, 1173104118, 458315668, 392670139, 1994471656, 860470682, 195502046, 446973189, 1634930755, 1869627248, 1539371366, 1496077548, 1900453330, 777675680, 165160302, 1474813612, 1338489492, 1976185787, 1443910983, 864076467, 374968648, 527958954, 1836984730, 429092808, 885126063, 1177501925, 1843659587, 608940199, 1051397497, 636638476, 1276611054, 17959668, 2127281294, 51965419, 1628418748, 310894343, 1151523377, 369117347, 1687537840, 711438217, 513298366, 686549514, 2136276614, 50992056, 2111616318, 1619851259, 2093901497, 769461890, 137863536, 695601311, 1344419594, 786566446, 1755062038, 1645604807, 1504010815, 477045502, 1277865998, 1156344676, 2012263489, 537860942, 2011637489, 1262102384, 662804163, 1933287234, 1242485210, 1903224533, 75763508, 723219588, 832695642, 1272723953, 1285408562, 1436749713, 1389806118, 1237787959, 1557748343, 602950949, 1310686824, 522630552, 1690758641, 1405756874, 447042107, 1476798943, 525532133, 1329560060, 1379026088, 1400303867, 1226067076, 777252286, 38621462, 704215756, 1806714156, 304163055, 226323479, 543619542, 1482495618, 278877779, 754339556, 1701936039, 1364931387, 1458486961, 964172482, 1468060987, 222144751, 1733093759, 1645219017, 2011073908, 1616334499, 1953615835, 1030439129, 1613602097, 2102783581, 358169077, 1347830114, 101828359, 1538824280, 850743511, 1610741967, 1892877570, 1373258743, 494449244, 299417772, 1290234665, 612124094, 979727234, 82968796, 1455944869, 1913457220, 434584433, 4624806, 975651142, 1550321466, 953112859, 1955231408, 186598089, 1226395504, 1063241475, 1153775060, 1157453973, 1373902028, 172154232, 235192290, 1434515890, 504778469, 2047436421, 1561907858, 1458188385, 588412639, 1656871975, 131409311, 166535635, 759832287, 1934582332, 231339949, 1767134791, 1610260169, 887099899, 695422452, 315479445, 546416606, 1895604474, 1327372681, 1273847650, 350353884, 1968587684, 1204349090, 1600133059, 2040815486, 695682824, 662183952, 1880768065, 1349345303, 1328780616, 1539940103, 839884361, 2094558551, 2126100970, 1210765454, 1037469446, 842943212, 76152554]


def compute_metrics(detected_drifts, real_drifts, duration, true_positive_cutoff=None, verbose=False):
    """
    Given the results from a drift detection method and the time points where the drifts actually happened, compute scores like true positives, false positives, ...
    :param detected_drifts: output from a drift detection method (each entry must be a tuple)
    :param real_drifts: time points where the drifts actually happened
    :param duration: how long the method took (this is just passed through this function)
    :param true_positive_cutoff:
    :param verbose: whether to print additional information
    :return: true positives, false positives, false negatives, delay sum, signed location error sum, and duration
    """
    if verbose:
        print("Took", duration, "seconds")
        print("detected drifts:", detected_drifts)
        print("real drifts:", real_drifts)
    true_positive_min_cutoff = 0
    drift_matching_matrix = np.zeros((len(detected_drifts), len(real_drifts)))
    # Causality (drifts can only be detected after they happened, and if a cutoff is given, that is applied here too)
    for i in range(len(detected_drifts)):
        for j in range(len(real_drifts)):
            if (real_drifts[j] + true_positive_min_cutoff) < detected_drifts[i][0] and (true_positive_cutoff is None or detected_drifts[i][0] < (real_drifts[j]+true_positive_cutoff))\
                    and abs(real_drifts[j]-detected_drifts[i][1]) < 50:  # If the estimated drift location is wrong by more than 50 samples, it can't be this drift
                drift_matching_matrix[i, j] = 1
    # Reporting a single actual drift
    for i in range(len(detected_drifts)):
        if np.sum(drift_matching_matrix[i, :]) <= 1:  # More than one drift?
            continue
        closest_actual_drift = -1
        for j in range(len(real_drifts)):
            if drift_matching_matrix[i, j] == 1:
                if closest_actual_drift == -1:
                    closest_actual_drift = j
                elif abs(real_drifts[j] - detected_drifts[i][1]) < abs(real_drifts[closest_actual_drift] - detected_drifts[i][1]):
                    closest_actual_drift = j
        drift_matching_matrix[i, :] = 0
        drift_matching_matrix[i, closest_actual_drift] = 1
    # Matching each real drift with the earliest detected drift
    for j in range(len(real_drifts)):
        if np.sum(drift_matching_matrix[:, j]) <= 1:  # More than one drift?
            continue
        closest_detected_drift = -1
        for i in range(len(detected_drifts)):
            if drift_matching_matrix[i, j] == 1:
                if closest_detected_drift == -1:
                    closest_detected_drift = i
                elif abs(real_drifts[j] - detected_drifts[i][1]) < abs(real_drifts[j] - detected_drifts[closest_detected_drift][1]):
                    closest_detected_drift = i
        drift_matching_matrix[:, j] = 0
        drift_matching_matrix[closest_detected_drift, j] = 1
    if verbose:
        print(drift_matching_matrix)
    nr_false_positives = 0
    nr_false_negatives = 0
    nr_true_positives = 0
    nr_true_positives2 = 0
    for j in range(drift_matching_matrix.shape[0]):
        if np.sum(drift_matching_matrix[j, :]) == 0:
            nr_false_positives += 1
        elif np.sum(drift_matching_matrix[j, :]) == 1:
            nr_true_positives += 1
        else:
            print("Error: np.sum(drift_matching_matrix[", j, ", :]) is", np.sum(drift_matching_matrix[j, :]), "should be 0 or 1")
            exit()
    for j in range(drift_matching_matrix.shape[1]):
        if np.sum(drift_matching_matrix[:, j]) == 0:
            nr_false_negatives += 1
        elif np.sum(drift_matching_matrix[:, j]) == 1:
            nr_true_positives2 += 1
        else:
            print("Error: np.sum(drift_matching_matrix[:, ", j, "]) is", np.sum(drift_matching_matrix[:, j]), "should be 0 or 1")
            exit()
    assert (nr_true_positives == nr_true_positives2)
    delay_sum = 0.0
    location_error_sum = 0.0
    signed_location_error_sum = 0.0
    denominator = 0
    for i in range(len(detected_drifts)):
        for j in range(len(real_drifts)):
            if drift_matching_matrix[i, j] == 1:
                if verbose:
                    print("Drift at sample", real_drifts[j], "reported with delay", detected_drifts[i][0] - real_drifts[j], ", diff to real drift time point", detected_drifts[i][1] - real_drifts[j])
                delay_sum += abs(detected_drifts[i][0] - real_drifts[j])  # abs is actually not necessary since drifts can only be detected after they happen
                location_error_sum += abs(detected_drifts[i][1] - real_drifts[j])
                signed_location_error_sum += (detected_drifts[i][1] - real_drifts[j])
                denominator += 1
    if verbose:
        print(nr_true_positives, "true positives")
        print(nr_false_positives, "false positives")
        print(nr_false_negatives, "false negatives")
    if denominator > 0:
        delay_sum /= denominator
        location_error_sum /= denominator
        signed_location_error_sum /= denominator
    else:
        delay_sum = np.nan
        location_error_sum = np.nan
        signed_location_error_sum = np.nan
    return nr_true_positives, nr_false_positives, nr_false_negatives, delay_sum, location_error_sum, signed_location_error_sum, duration


class DetectorResult:
    def __init__(self, detector_name: str, parameters: dict, false_negatives, false_positives, detection_delays, location_errors, run_times, true_positives, signed_location_errors):
        self.detection_name = detector_name
        self.parameters = parameters
        self.false_negatives = np.nanmean(false_negatives)
        self.false_positives = np.nanmean(false_positives)
        self.true_positives = np.nanmean(true_positives)
        if self.true_positives != 0.0:
            self.detection_delays = np.nanmean(detection_delays)
            self.location_errors = np.nanmean(location_errors)
            self.signed_location_errors = np.nanmean(signed_location_errors)
            self.detection_delays_std = np.nanstd(detection_delays)
            self.location_errors_std = np.nanstd(location_errors)
            self.signed_location_errors_std = np.nanstd(signed_location_errors)
        else:
            self.detection_delays = np.nan
            self.location_errors = np.nan
            self.signed_location_errors = np.nan
            self.detection_delays_std = np.nan
            self.location_errors_std = np.nan
            self.signed_location_errors_std = np.nan
        self.run_time = np.nanmean(run_times)
        self.false_negatives_std = np.nanstd(false_negatives)
        self.false_positives_std = np.nanstd(false_positives)
        self.true_positives_std = np.nanstd(true_positives)
        self.run_time_std = np.nanstd(run_times)
        self.nr_of_runs = min(len(run_times), len(location_errors))


def run_and_evaluate_one(dataset_name: str, detector_name: str, params, my_seed, visualize=False, verbose=False):
    """
    Run and evaluate a specific parameter configuration for one specific seed
    :param dataset_name: name of the dataset, e.g. mnist, rialto, covtype, or music
    :param detector_name: for example MMDDDM, HDDDM, or D3
    :param params: the parameters for the drift detection method
    :param my_seed: the random seed to use
    :param visualize: whether to do visualizations when a drift is detected
    :param verbose: whether to print additional information
    :return:
    """
    random.seed(my_seed)
    if dataset_name == "trivial":
        data, real_drifts = trivial_dataset(length=1600, drifts=2)
    elif dataset_name == "mnist":
        data, real_drifts = mnist_dataset(length=1600, drifts=2)
    elif dataset_name == "rialto":
        data, real_drifts = rialto_dataset(length=1600, drifts=2)
    elif dataset_name == "covtype":
        data, real_drifts = covtype_dataset(length=1600, drifts=2)
    elif dataset_name == "music":
        data, real_drifts = music_dataset(length=1600, drifts=2)
    else:
        print("ERROR: Unknown dataset name", dataset_name)
        exit()
    if detector_name == "HDDDM":
        multifeature_detector = HDDDM(**params,
                                      visualize=visualize, verbose=verbose)
        true_positive_cutoff = None
    elif detector_name == "MMDDDM":
        multifeature_detector = MMDDDM(**params,
                                       visualize=visualize, verbose=verbose)
        true_positive_cutoff = None
    elif detector_name == "D3":
        multifeature_detector = D3(**params,
                                   visualize=visualize, verbose=verbose
                                   )
        true_positive_cutoff = params["window_size"]
    elif detector_name == "SpectralDDM":
        multifeature_detector = SpectralDDM(**params,
                                            )
        true_positive_cutoff = multifeature_detector.max_window_size
    else:
        print("Unknown detector")
        exit()
    detected_drifts = []
    start = time()
    for i in range(data.shape[0]):
        if verbose and (i-0.5) in real_drifts:
            print("Drift happened at", i-0.5)
        result = multifeature_detector.update(data[i, :])
        for res in result:
            detected_drifts.append((i, i - res))
    end = time()
    return compute_metrics(detected_drifts, real_drifts, end - start, true_positive_cutoff=true_positive_cutoff, verbose=verbose)

def run_and_evaluate(dataset_name: str, detector_name: str, nr_of_runs: int, params, parallel=True, visualize=False, verbose=False):
    """
    Run and evaluate a specific parameter configuration with many different seeds
    :param dataset_name: name of the dataset, e.g. mnist, rialto, covtype, or music
    :param detector_name: for example MMDDDM, HDDDM, or D3
    :param nr_of_runs: for how many seeds should each parameter configuration be run
    :param params: the parameters for the drift detection method
    :param parallel: whether to do the computation in parallel (multiple processes)
    :param visualize: whether to do visualizations when a drift is detected
    :param verbose: whether to print additional information
    :return: the metrics (true positives, false positives, false negatives, ...)
    """
    all_true_positives = np.zeros(nr_of_runs)
    all_false_positives = np.zeros(nr_of_runs)
    all_false_negatives = np.zeros(nr_of_runs)
    all_detection_delays = np.zeros(nr_of_runs)
    all_location_errors = np.zeros(nr_of_runs)
    all_signed_location_errors = np.zeros(nr_of_runs)
    all_run_times = np.zeros(nr_of_runs)

    print("Running", dataset_name, ",", detector_name, ", params=", params)
    if parallel and nr_of_runs > 1 and not visualize and not verbose:
        # Parallel
        results_waitlist = []
        with Pool(processes=None) as pool:
            for run in range(nr_of_runs):
                my_seed = seeds_for_runs[run]
                results_waitlist.append(pool.apply_async(func=run_and_evaluate_one, kwds={"dataset_name": dataset_name, "detector_name": detector_name, "params": params, "my_seed": my_seed}))
            for run in range(nr_of_runs):
                all_true_positives[run], \
                    all_false_positives[run], \
                    all_false_negatives[run], \
                    all_detection_delays[run], \
                    all_location_errors[run], \
                    all_signed_location_errors[run],\
                    all_run_times[run] = results_waitlist[run].get(timeout=None)
    else:
        # Sequential
        for run in range(nr_of_runs):
            my_seed = seeds_for_runs[run]
            all_true_positives[run], \
                all_false_positives[run], \
                all_false_negatives[run], \
                all_detection_delays[run], \
                all_location_errors[run], \
                all_signed_location_errors[run], \
                all_run_times[run] = run_and_evaluate_one(dataset_name=dataset_name, detector_name=detector_name, params=params, my_seed=my_seed, visualize=visualize, verbose=verbose)
    return all_true_positives, all_false_positives, all_false_negatives, all_detection_delays, all_location_errors, all_signed_location_errors, all_run_times


def prune_pareto(pareto_frontier, cmp):
    """
    Prune the pareto frontier, that means: remove solutions where another solution is better in every dimension
    :param pareto_frontier: the pareto frontier that is pruned (this is also used for output and will thus be altered!)
    :param cmp: comparator that says whether an item can be removed (cmp should return true if the second argument is worse than the first and can be removed)
    """
    i = 0
    while i < len(pareto_frontier):
        j = 0
        while j < len(pareto_frontier):
            if i != j and cmp(pareto_frontier[i], pareto_frontier[j]):
                print("Removing:", pareto_frontier[j].parameters, "pareto_frontier[i].fn=", pareto_frontier[i].false_negatives, "pareto_frontier[j].fn=", pareto_frontier[j].false_negatives,
                      "pareto_frontier[i].fp=", pareto_frontier[i].false_positives, "pareto_frontier[j].fp=", pareto_frontier[j].false_positives)
                pareto_frontier.pop(j)
                i = 0
                j = 0
            else:
                j += 1
        i += 1


def compute_pareto_frontier(dataset_name: str, detector_name: str, nr_of_runs: int = 200, exploring_iterations: int = 500, localize_drifts=True, stride1=False, verbose=False, visualize=False, parallel=True):
    """
    Compute/Refine the pareto frontier by loading/storing a pickle file
    :param dataset_name: name of the dataset, e.g. mnist, rialto, covtype, or music
    :param detector_name: for example MMDDDM, HDDDM, or D3
    :param nr_of_runs: for how many seeds should each parameter configuration be run
    :param exploring_iterations: how many different parameter configurations should be tested?
    :param localize_drifts: whether to localize the drifts in time, as described in "Extending Drift Detection Methods to Identify When Exactly the Change Happened" by Vieth et al.
    :param stride1: whether a stride of 1 should be used (whether the drift test should be run for every sample, if the method supports this)
    :param verbose: whether to print additional information
    :param visualize: whether to do visualizations when a drift is detected
    :param parallel: whether to do the computation in parallel (multiple processes)
    """
    if localize_drifts:
        if stride1:
            suffix = "_E"
        else:
            suffix = ""
    else:
        if stride1:
            suffix = "_NE"
        else:
            suffix = "_N"
    pickle_filename = detector_name + suffix + "_" + dataset_name + "_pareto_frontier.pickle"
    if detector_name == "HDDDM":
        params_to_optimize = ["batching_size", "gamma"]
    elif detector_name == "MMDDDM":
        params_to_optimize = ["batching_size", "gamma"]
    elif detector_name == "D3":
        params_to_optimize = ["window_size", "auc_threshold"]
    elif detector_name == "SpectralDDM":
        params_to_optimize = ["batch_distance", "n_eigen", "max_possible_number_splits", "n_splits", "test_size", "min_samples_per_drift", "min_window_size", "max_window_size"]
    else:
        exit()
    pareto_frontier = []
    try:
        with open(pickle_filename, "rb") as f:
            initialization = pickle.load(file=f)
    except FileNotFoundError:
        print("Couldn't find", pickle_filename, ", trying again with", (detector_name + "_" + dataset_name + "_pareto_frontier.pickle"))
        try:
            with open(detector_name + "_" + dataset_name + "_pareto_frontier.pickle", "rb") as f:
                initialization = pickle.load(file=f)
        except FileNotFoundError:
            print("Generating an initial parameter set ...")
            initialization = []
            if detector_name == "HDDDM":
                for batching_size in [65, 85]:
                    for gamma in [1.0, 4.0]:
                        initialization.append(DetectorResult(detector_name=detector_name, parameters={"batching_size": batching_size, "gamma": gamma}, false_negatives=np.ones(nr_of_runs),
                                                             false_positives=np.ones(nr_of_runs), detection_delays=np.ones(nr_of_runs), location_errors=np.ones(nr_of_runs),
                                                             run_times=np.ones(nr_of_runs), true_positives=np.ones(nr_of_runs), signed_location_errors=np.ones(nr_of_runs)))
            elif detector_name == "MMDDDM":
                for batching_size in [55, 75]:
                    for gamma in [0.003, 0.07]:
                        initialization.append(DetectorResult(detector_name=detector_name, parameters={"batching_size": batching_size, "gamma": gamma}, false_negatives=np.ones(nr_of_runs),
                                                             false_positives=np.ones(nr_of_runs), detection_delays=np.ones(nr_of_runs), location_errors=np.ones(nr_of_runs),
                                                             run_times=np.ones(nr_of_runs), true_positives=np.ones(nr_of_runs), signed_location_errors=np.ones(nr_of_runs)))
            elif detector_name == "D3":
                for window_size in [185, 225]:
                    for auc_threshold in [0.66, 0.95]:
                        initialization.append(DetectorResult(detector_name=detector_name, parameters={"window_size": window_size, "auc_threshold": auc_threshold}, false_negatives=np.ones(nr_of_runs),
                                                             false_positives=np.ones(nr_of_runs), detection_delays=np.ones(nr_of_runs), location_errors=np.ones(nr_of_runs),
                                                             run_times=np.ones(nr_of_runs), true_positives=np.ones(nr_of_runs), signed_location_errors=np.ones(nr_of_runs)))
            elif detector_name == "SpectralDDM":
                initialization.append(DetectorResult(detector_name=detector_name, parameters={"batch_distance": 90, "n_eigen": 15, "max_possible_number_splits": 5, "n_splits": 20, "test_size": 0.4,
                                                                                              "min_samples_per_drift": 10, "min_window_size": 40, "max_window_size": 500},
                                                     false_negatives=np.ones(nr_of_runs), false_positives=np.ones(nr_of_runs), detection_delays=np.ones(nr_of_runs),
                                                     location_errors=np.ones(nr_of_runs), run_times=np.ones(nr_of_runs), true_positives=np.ones(nr_of_runs), signed_location_errors=np.ones(nr_of_runs))
                                      )
            else:
                exit()
    prune_pareto(initialization, lambda item1, item2: ((item1.false_negatives < item2.false_negatives and item1.false_positives < item2.false_positives) or
                                                       (item1.false_negatives <= item2.false_negatives and item1.false_positives == 0.0 and item2.false_positives == 0.0) or
                                                       (item1.false_positives <= item2.false_positives and item1.false_negatives == 0.0 and item2.false_negatives == 0.0)))
    print("Initial pareto frontier:")
    for p in initialization:
        print(p.parameters, ",  #", "fp=", p.false_positives, "fn=", p.false_negatives, p.detection_delays, p.location_errors, p.run_time)
    for i in range(len(initialization)):
        if isinstance(initialization[i].parameters, dict):
            params_dict = initialization[i].parameters
        else:
            params_dict = {}
            if detector_name == "HDDDM":
                params_dict["batching_size"] = int(initialization[i].parameters[0])
                params_dict["gamma"] = initialization[i].parameters[1]
            elif detector_name == "MMDDDM":
                params_dict["batching_size"] = int(initialization[i].parameters[0])
                params_dict["gamma"] = initialization[i].parameters[1]
            elif detector_name == "D3":
                params_dict["window_size"] = int(initialization[i].parameters[0])
                params_dict["auc_threshold"] = initialization[i].parameters[1]
            elif detector_name == "SpectralDDM":
                params_dict["batch_distance"] = int(initialization[i].parameters[0])
                params_dict["n_eigen"] = int(initialization[i].parameters[1])
                params_dict["max_possible_number_splits"] = 10
                params_dict["n_splits"] = 20
                params_dict["test_size"] = 0.45
                params_dict["min_samples_per_drift"] = 10
                params_dict["min_window_size"] = 50
                params_dict["max_window_size"] = 500
            else:
                exit()
        params_dict["localize_drifts"] = localize_drifts
        if stride1:
            params_dict["stride"] = 1
        else:
            params_dict["stride"] = None
        true_positives, false_positives, false_negatives, detection_delays, location_errors, signed_location_errors, run_times = \
            run_and_evaluate(dataset_name=dataset_name, detector_name=detector_name, nr_of_runs=nr_of_runs, params=params_dict, parallel=parallel, verbose=verbose, visualize=visualize)
        pareto_frontier.append(DetectorResult(detector_name=detector_name, parameters=params_dict, false_negatives=false_negatives, false_positives=false_positives, detection_delays=detection_delays, location_errors=location_errors, run_times=run_times, true_positives=true_positives, signed_location_errors=signed_location_errors))
    random.shuffle(pareto_frontier)  # sorted is stable, which we don't necessarily want here
    pareto_frontier = sorted(pareto_frontier, key=cmp_to_key(lambda item1, item2: item2.false_positives - item1.false_positives))
    pareto_frontier = sorted(pareto_frontier, key=cmp_to_key(lambda item1, item2: item1.false_negatives - item2.false_negatives))
    # main part
    for k in range(exploring_iterations):
        print("compute_pareto_frontier: iteration", k, "of", exploring_iterations, ", pareto frontier has", len(pareto_frontier), "elements")
        # New parameters
        if k % 2 == 0:
            where = random.random()*min(pareto_frontier[-1].false_negatives, 2)
            randnr = 0
            while randnr < len(pareto_frontier) and where > pareto_frontier[randnr].false_negatives and (randnr+1) <= len(pareto_frontier):
                randnr += 1
        else:
            where = random.random()*min(pareto_frontier[0].false_positives, 2)
            randnr = 0
            while randnr < len(pareto_frontier) and where < pareto_frontier[randnr].false_positives and (randnr+1) <= len(pareto_frontier):
                randnr += 1
        if randnr == 0:
            new_param = deepcopy(pareto_frontier[0].parameters)
        elif randnr == len(pareto_frontier):
            new_param = deepcopy(pareto_frontier[-1].parameters)
        else:
            new_param = {}
            for key in params_to_optimize:
                if pareto_frontier[randnr-1].parameters[key] is None or pareto_frontier[randnr].parameters[key] is None:
                    new_param[key] = None
                else:
                    new_param[key] = type(pareto_frontier[randnr].parameters[key])(0.5*(pareto_frontier[randnr-1].parameters[key]+pareto_frontier[randnr].parameters[key]))
        for key in params_to_optimize:
            if new_param[key] is None:
                pass
            elif isinstance(new_param[key], int):
                new_param[key] += random.randint(-3, 3)
            else:
                new_param[key] *= (1+1.0*(random.random()-0.5))
        new_param["localize_drifts"] = localize_drifts
        if stride1:
            new_param["stride"] = 1
        else:
            new_param["stride"] = None
        true_positives, false_positives, false_negatives, detection_delays, location_errors, signed_location_errors, run_times = \
            run_and_evaluate(dataset_name=dataset_name, detector_name=detector_name, nr_of_runs=nr_of_runs, params=new_param, parallel=parallel, verbose=verbose, visualize=visualize)
        pareto_frontier.append(DetectorResult(detector_name=detector_name, parameters=new_param, false_negatives=false_negatives, false_positives=false_positives, detection_delays=detection_delays, location_errors=location_errors, run_times=run_times, true_positives=true_positives, signed_location_errors=signed_location_errors))
        random.shuffle(pareto_frontier)  # sorted is stable, which we don't necessarily want here
        pareto_frontier = sorted(pareto_frontier, key=cmp_to_key(lambda item1, item2: item2.false_positives - item1.false_positives))
        pareto_frontier = sorted(pareto_frontier, key=cmp_to_key(lambda item1, item2: item1.false_negatives - item2.false_negatives))

        prune_pareto(pareto_frontier, lambda item1, item2: ((item1.false_negatives < item2.false_negatives and item1.false_positives < item2.false_positives) or
                                                            (item1.false_negatives <= item2.false_negatives and item1.false_positives == 0.0 and item2.false_positives == 0.0) or
                                                            (item1.false_positives <= item2.false_positives and item1.false_negatives == 0.0 and item2.false_negatives == 0.0)))
        with open(pickle_filename, "wb") as f:
            pickle.dump(pareto_frontier, f)
    dim1 = []
    dim2 = []
    print("Final pareto frontier:")
    for p in pareto_frontier:
        print(p.parameters, ",  #", "fp=", p.false_positives, "fn=", p.false_negatives, p.detection_delays, p.location_errors, p.run_time)
        dim1.append(p.false_negatives)
        dim2.append(p.false_positives)
    print("plt.plot(", dim1, ",", dim2, ")")


if __name__ == "__main__":
    nr_of_runs = 250
    parallel = True
    visualize = False
    verbose = False
    localize_drifts = True
    stride1 = False
    args = sys.argv[1:]
    if len(args) == 0:
        print("Possible keywords are: HDDDM, MMDDDM, D3, SpectralDDM, covtype, mnist, rialto, music, visualize, verbose")
    for arg in args:
        if arg[-4:] == "runs":
            nr_of_runs = int(arg[:-4])
            args.remove(arg)
            break
    test_methods = []
    if "HDDDM" in args:
        test_methods.append("HDDDM")
        args.remove("HDDDM")
    if "MMDDDM" in args:
        test_methods.append("MMDDDM")
        args.remove("MMDDDM")
    if "D3" in args:
        test_methods.append("D3")
        args.remove("D3")
    if "SpectralDDM" in args:
        test_methods.append("SpectralDDM")
        args.remove("SpectralDDM")
    test_datasets = []
    if "covtype" in args:
        test_datasets.append("covtype")
        args.remove("covtype")
    if "rialto" in args:
        test_datasets.append("rialto")
        args.remove("rialto")
    if "mnist" in args:
        test_datasets.append("mnist")
        args.remove("mnist")
    if "music" in args:
        test_datasets.append("music")
        args.remove("music")
    if "trivial" in args:
        test_datasets.append("trivial")
        args.remove("trivial")
    if "visualize" in args:
        visualize = True
        args.remove("visualize")
    if "verbose" in args:
        verbose = True
        args.remove("verbose")
    if "serial" in args:
        parallel = False
        args.remove("serial")
    if "stride1" in args:
        stride1 = True
        args.remove("stride1")
    if "nolocalizedrifts" in args:
        localize_drifts = False
        args.remove("nolocalizedrifts")
    if len(args) > 0:
        print("unknown args: ", args)
        exit()
    # Parameter search
    for detector_name in test_methods:
        for dataset_name in test_datasets:
            compute_pareto_frontier(detector_name=detector_name, dataset_name=dataset_name, nr_of_runs=nr_of_runs, exploring_iterations=100,
                                    localize_drifts=localize_drifts, stride1=stride1, verbose=verbose, visualize=visualize, parallel=parallel)
