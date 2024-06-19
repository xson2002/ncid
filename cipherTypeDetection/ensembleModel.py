import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cipherTypeDetection.config as config
from cipherTypeDetection.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
from cipherImplementations.cipher import OUTPUT_ALPHABET
from util.utils import get_model_input_length


f1_ffnn = [0.85802469, 0.71601017, 1., 0.99176076, 0.2358263, 0.77622378, 0.70844478, 0.96519285, 0.94650687, 0.85040917, 0.70607827, 0.78005115, 0.84944426, 0.99756839, 0.97485951, 0.95828066, 0.94724363, 0.9211165, 0.99908898, 0.95622688, 0.99787557, 0.74629137, 0.93963723, 0.50773994, 0.95047098, 0.99098016, 0.81281534, 1., 0.99939357, 0.94468614, 0.74026438, 0.80489161, 0.99756691, 0.98346859, 0.96463596, 0.54026417, 0.98918919, 0.2, 0.11942675, 0.14091471, 0.14470678, 0.12706072, 0.99969651, 0.55685131, 0.0059312, 0.99969669, 0.68258591, 0.79946255, 0.49982462, 0.94923858, 0.78894205, 0.8, 0.95987842, 0.87563309, 0.20748299, 0.11743451, 0.04731679, 0.41069723, 0.32477514, 0.11273486, 0.16886064]
f1_transformer = [0.03043478, 0.69008264, 1., 0.99455206, 0.58704225, 0.73749569, 0.73001076, 0.95184968, 0.57126168, 0.3715415, 0.7044335, 0.86148552, 0.9545597, 0.99301973, 0.98642534, 0.92614159, 0.99266952, 0.91793838, 0.99908898, 0.91535492, 0.98880484, 0.9969697, 0.98505642, 0.11973019, 0.3258499, 0.99817961, 0.82288401, 1., 0.99969651, 0.99516616, 0.77780972, 0.7008872, 0.49159664, 0.60615959, 0.91304348, 0.79020173, 0.97655768, 0.2, 0.1637931, 0.22581529, 0.18894137, 0.09254089, 1., 0.15538462, 0.27428112, 0.96774194, 0.6044648, 0.48468399, 0.68386023, 0.52110977, 0.99574985, 0.84159378, 0.95633188, 0.87079446, 0.29268293, 0.33854521, 0.04137931, 0.38119935, 0.24557878, 0.06648794, 0.15943396]
f1_lstm = [0.87462277, 0.62320649, 1., 0.98056801, 0.1368301, 0.52678571, 0.56427256, 0.67673716, 0.67601582, 0.28803658, 0.51890034, 0.74213075, 0.86412023, 0.98335855, 0.95962551, 0.99454876, 0.99178582, 0.90378833, 0.99878493, 0.89454763, 0.8844102, 0.95705148, 0.92845887, 0.38333333, 0.43565574, 0.99246307, 0.77978339, 1., 1., 0.99213075, 0.66302368, 0.58091286, 0.99939357, 0.70704574, 0.90360333, 0.65119652, 0.96410561, 0.2, 0.07527802, 0.12055016, 0.08519982, 0.04906667, 1., 0.37467192, 0.44055467, 1., 0.53605096, 0.56418384, 0.64847059, 0.95567699, 0.91383495, 0.73835784, 0.73876404, 0.77348066, 0.22491787, 0.08076729, 0.03540834, 0.38360551, 0.17638426, 0.03544883, 0.1492823]
f1_rf = [0.8785558, 0.68663839, 0.99969651, 0.99516908, 0.16649467, 0.75178998, 0.76286509, 0.99328039, 0.92298716, 0.80948698, 0.71287739, 0.82899278, 0.88748875, 0.99305765, 0.99788071, 0.95795131, 0.98313253, 0.81518066, 0.99969651, 0.99666969, 0.9963548, 0.99756691, 0.99604743, 0.42259595, 0.90221351, 0.9993932, 0.82558452, 1., 1., 0.98288509, 0.80442338, 0.82807431, 0.99424068, 0.96756757, 0.99696049, 0.47208855, 0.97886276, 0.2, 0.11394617, 0.12567935, 0.12958963, 0.10682064, 0.99969651, 0.43733044, 0.41083261, 1., 0.53218355, 0.82228117, 0.60863605, 0.98339873, 0.99545867, 0.836934, 0.98180716, 0.8815749, 0.14111072, 0.15636105, 0.04891923, 0.35687981, 0.33549064, 0.15529179, 0.18460153]
f1_nb = [0.43368752, 0.49722479, 1., 0.89344019, 0.03794872, 0.4570169, 0.41788144, 0.74102871, 0.46397807, 0.14561196, 0.40013228, 0.65718608, 0.6151569, 0.94468887, 0.96742002, 0.79916939, 0.8866146, 0.63759938, 0.99908898, 0.7658684, 0.78235294, 0.66558704, 0.62588431, 0.10453649, 0.18856655, 0.97313257, 0.72310498, 1., 0.98771352, 0.85376867, 0.43097643, 0.48707753, 0.61706485, 0.88359502, 0.62966031, 0.22506562, 0.80921758, 0.2, 0.04012036, 0.03501751, 0.03508772, 0.02456499, 1., 0.37869225, 0.08797032, 0.75560538, 0.19837195, 0.43834772, 0.19837977, 0.72571617, 0.62120886, 0.6510913, 0.60896767, 0.48095238, 0.02794411, 0.04343534, 0.02680653, 0.28351955, 0.22888514, 0.12357217, 0.13593203]
accuracy_ffnn = [0.9954234320280165, 0.9911055396370583, 1.0, 0.9997313753581661, 0.9747890799108564, 0.9917223814071952, 0.9924088666029927, 0.9988956542502387, 0.9981793218720153, 0.9954532792104426, 0.9894639446036294, 0.9931550461636421, 0.9955527698185291, 0.9999204075135307, 0.999154329831264, 0.998686723973257, 0.9982290671760585, 0.9974132441897485, 0.9999701528175741, 0.9985872333651703, 0.9999303565743394, 0.993363976440624, 0.9979803406558421, 0.985762893982808, 0.9983783030881884, 0.9997015281757402, 0.9926177968799745, 1.0, 0.9999801018783827, 0.9982290671760585, 0.9927670327921044, 0.9930157593123209, 0.9999204075135307, 0.9994528016555237, 0.9988160617637695, 0.9785299267749125, 0.9996418338108882, 0.9, 0.9779926774912449, 0.9792562082139447, 0.9776544094237504, 0.9784005889843999, 0.9999900509391914, 0.9758038841133397, 0.9833253740846865, 0.9999900509391914, 0.989155523718561, 0.9940604106972302, 0.9716252785737026, 0.9984081502706145, 0.9921004457179242, 0.9931450971028335, 0.998686723973257, 0.9956025151225725, 0.9721824259789876, 0.980559535179879, 0.9835741006049029, 0.9815842884431709, 0.9589203279210442, 0.9830865966252785, 0.9817832696593441]
accuracy_transformer = [0.9822514052628961, 0.9895537979406058, 1.0, 0.9998209222504104, 0.9854151121723126, 0.9924289906979058, 0.9925085808088345, 0.9983684027259613, 0.9853952146445805, 0.9699447843605432, 0.990449186688554, 0.9951947470526787, 0.998557429239417, 0.99977117843108, 0.9995523056260259, 0.9976023479082724, 0.9997612296672138, 0.9971646022981645, 0.9999701537084017, 0.9971646022981645, 0.9996318957369547, 0.9999005123613391, 0.9995125105705616, 0.9792269810476049, 0.9790876983534795, 0.9999403074168035, 0.9932547380987913, 1.0, 0.9999900512361339, 0.9998408197781425, 0.9923096055315127, 0.9906083669104114, 0.9807391931552505, 0.9832064865940406, 0.9972143461174949, 0.9927572999054868, 0.9992438939461772, 0.9, 0.9710490971496791, 0.9718947420782967, 0.9709595582748843, 0.9773665622046461, 1.0, 0.9781525145500671, 0.9640949112072825, 0.9989752773217928, 0.9862508083370641, 0.9867780928219668, 0.9886584091926578, 0.980251703725812, 0.9998607173058748, 0.9951350544694821, 0.9986071730587475, 0.995453414913197, 0.979804009351838, 0.9806397055165895, 0.9834054618713625, 0.9810078097796349, 0.9626523404466995, 0.982679202109138, 0.9822713027906282]
accuracy_lstm = [0.9954535958376028, 0.9879823714919568, 1.0, 0.9993533560819348, 0.9774072563395975, 0.985236621932172, 0.9882310806912126, 0.9904197216446642, 0.9910365204588187, 0.9721147245794327, 0.9860722848416718, 0.9915239904893602, 0.9956824083009183, 0.9994528397616371, 0.9986271252201077, 0.9998209293765358, 0.9997313940648037, 0.996816522249525, 0.999960206528119, 0.9964981744744774, 0.9961201364916086, 0.998597280116197, 0.9977417204707568, 0.9860125946338503, 0.9726021946099742, 0.9997512908007441, 0.9908972433072355, 1.0, 1.0, 0.9997413424327739, 0.9871168634785463, 0.9869377928550821, 0.9999801032640595, 0.9905689471642177, 0.9968861608253166, 0.9856445050189516, 0.9987962474756016, 0.9, 0.978491628448353, 0.9783722480327103, 0.9792775495180016, 0.9822620599090719, 1.0, 0.9810384106487331, 0.9755170664252529, 1.0, 0.981884021926203, 0.9863209940409275, 0.9851371382524697, 0.9985475382763458, 0.9971746634964535, 0.9915040937534197, 0.990748017787682, 0.9914344551776281, 0.9694883554352909, 0.981884021926203, 0.9831972064982739, 0.9808493916572987, 0.9385887245197425, 0.9832171032342144, 0.9823118017489231]
accuracy_rf = [0.9955827048968303, 0.9882404441172374, 0.999990051137155, 0.9998408181944803, 0.9758939053266211, 0.9917225461129793, 0.9932148755397258, 0.9997811250174105, 0.9973735002089261, 0.9926477903575621, 0.9907276598284817, 0.9945778697494876, 0.9962691764331337, 0.9997711761545656, 0.9999303579600851, 0.9986768012416181, 0.9994428636806812, 0.9929263585172214, 0.999990051137155, 0.9998905625087052, 0.9998806136458602, 0.9999204090972402, 0.9998706647830153, 0.9866784726505761, 0.996527846867103, 0.99998010227431, 0.9935431880136101, 1.0, 1.0, 0.9994428636806812, 0.9938416538989594, 0.9941998129613785, 0.9998109716059455, 0.9989255228127425, 0.9999005113715502, 0.9781721949181209, 0.9992936307380067, 0.9, 0.9747796326879837, 0.9743916270370296, 0.975943649640846, 0.9753765644586824, 0.999990051137155, 0.9793660584595181, 0.9796545754820224, 1.0, 0.9848876773384803, 0.994000835704479, 0.9829575979465547, 0.9994528125435262, 0.9998507670573253, 0.9945181765724178, 0.9994030682293014, 0.996169687804684, 0.975535746264202, 0.976381399606025, 0.9833655013231988, 0.9808882344747996, 0.9622042700519331, 0.9830073422607796, 0.981982609387747]
accuracy_nb = [0.9856840136494325, 0.9783718177024782, 1.0, 0.9964284648367938, 0.9813364903449168, 0.9852960195787778, 0.9761632360695206, 0.9913845419182825, 0.9727608265268561, 0.9784016634002208, 0.9819533014315986, 0.9887282748191848, 0.9870668643115095, 0.9982490523990967, 0.9988957091835212, 0.9932648208760707, 0.9958116537501119, 0.9859426763632022, 0.9999701543022573, 0.9899817941243769, 0.9941104489787798, 0.9917824845548514, 0.9910562392431131, 0.9819334042997702, 0.9810778276311469, 0.9990946805018056, 0.9894246744331805, 1.0, 0.9995921087975168, 0.9958116537501119, 0.9848682312444661, 0.9794661599530428, 0.9888377090442413, 0.9957221166568839, 0.9911059820726842, 0.9530029746212083, 0.9929166210690729, 0.9, 0.9809584448401762, 0.9808092163514629, 0.9808490106151199, 0.9810380333674901, 1.0, 0.975516579285096, 0.9828785180616214, 0.9934936378920979, 0.9363192295830556, 0.9818737129042848, 0.9596386680859954, 0.9917128445934519, 0.9824805754250525, 0.9887083776873564, 0.9850771511286648, 0.9783121263069928, 0.9806201935990927, 0.9807196792582349, 0.9833858949232468, 0.9795855427440134, 0.9636678372812559, 0.9832068207367908, 0.9827989295343076]
recall_ffnn = [0.8434466, 0.68385922, 1., 0.98604369, 0.23725728, 0.8756068, 0.56801471, 0.93385922, 0.98240291, 0.78822816, 0.77184466, 0.74029126, 0.7651699, 0.99575243, 1., 0.91990291, 0.96966019, 0.9211165, 0.99817961, 0.94114078, 0.99757282, 0.59526699, 0.95873786, 0.44781553, 0.94902913, 1., 0.97754854, 1., 1., 0.9223301, 0.62864078, 0.87864078, 0.99514563, 0.99271845, 0.9848301, 0.76941748, 0.9993932, 0.2, 0.09101942, 0.10376214, 0.11529126, 0.09587379, 0.9993932, 0.92718447, 0.00303398, 1., 0.71116505, 0.72208738, 0.86468447, 0.90776699, 0.90048544, 0.83616505, 0.95813107, 0.94417476, 0.22208738, 0.0788835, 0.02487864, 0.3913835, 0.60254854, 0.06553398, 0.11286408]
recall_transformer = [0.01699029, 0.70934466, 1., 0.99696602, 0.63228155, 0.64866505, 0.62262997, 0.9836165, 0.5934466, 0.54186893, 0.69417476, 0.91140777, 0.92415049, 0.99271845, 0.99211165, 0.91686893, 0.98604369, 0.96723301, 0.99817961, 0.93507282, 0.99150485, 0.99817961, 0.97997573, 0.08616505, 0.30825243, 0.99817961, 0.95570388, 1., 0.9993932, 0.9993932, 0.82099515, 0.6711165, 0.56796117, 0.78822816, 0.89199029, 0.83191748, 0.96055825, 0.2, 0.17293689, 0.25, 0.20631068, 0.07038835, 1., 0.12257282, 0.41383495, 0.9375, 0.6407767, 0.37924757, 0.74817961, 0.65533981, 0.99514563, 0.78822816, 0.93021845, 0.93446602, 0.25485437, 0.30218447, 0.02184466, 0.35679612, 0.37075243, 0.03762136, 0.10254854]
recall_lstm = [0.96723301, 0.60618932, 1., 0.99514563, 0.1092233, 0.50121359, 0.46735815, 0.61165049, 0.57038835, 0.3440534, 0.45813107, 0.74393204, 0.83737864, 0.98604369, 0.99514563, 0.99635922, 0.98907767, 0.91201456, 0.99757282, 0.9059466, 0.90533981, 0.9532767, 0.89381068, 0.2651699, 0.64502427, 0.99878641, 0.98300971, 1., 1., 0.99453883, 0.77305825, 0.55218447, 1., 0.69417476, 0.8901699, 0.81735437, 0.98604369, 0.2, 0.05339806, 0.09041262, 0.05885922, 0.02791262, 1., 0.34648058, 0.58798544, 1., 0.63834951, 0.54004854, 0.83616505, 0.95509709, 0.91383495, 0.73118932, 0.79793689, 0.89199029, 0.27002427, 0.04854369, 0.01881068, 0.36347087, 0.40109223, 0.01881068, 0.09466019]
recall_rf = [0.97451456, 0.78580097, 0.9993932, 1., 0.14684466, 0.76456311, 0.67135863, 0.98665049, 0.95995146, 0.9526699, 0.70206311, 0.80157767, 0.89745146, 0.99817961, 1., 0.91929612, 0.99029126, 0.95145631, 0.9993932, 0.99878641, 0.99514563, 0.99514563, 0.99393204, 0.2973301, 0.97694175, 0.9993932, 0.93203883, 1., 1., 0.97572816, 0.77245146, 0.85194175, 0.99514563, 0.97754854, 0.99514563, 0.59526699, 0.99757282, 0.2, 0.09890777, 0.11225728, 0.1092233, 0.08980583, 0.9993932, 0.48907767, 0.43264563, 1., 0.52427184, 0.84648058, 0.80825243, 0.98847087, 0.99757282, 0.85800971, 0.98240291, 0.86953883, 0.12257282, 0.13349515, 0.02609223, 0.32342233, 0.58191748, 0.09526699, 0.1243932]
recall_nb = [0.33434466, 0.65230583, 1., 0.91322816, 0.02245146, 0.37742718, 0.52535125, 0.75182039, 0.7190534, 0.11225728, 0.36711165, 0.65898058, 0.63046117, 0.91201456, 1., 0.81735437, 0.99878641, 0.75424757, 0.99817961, 0.9993932, 0.64563107, 0.49878641, 0.45631068, 0.06432039, 0.13410194, 1., 0.84223301, 1., 1., 0.74575243, 0.34951456, 0.59466019, 0.54854369, 0.99029126, 0.46116505, 0.41626214, 0.91626214, 0.2, 0.02427184, 0.02123786, 0.02123786, 0.01456311, 1., 0.45509709, 0.05036408, 0.61347087, 0.48058252, 0.43143204, 0.30461165, 0.66868932, 0.87621359, 0.64259709, 0.70873786, 0.61286408, 0.01699029, 0.02669903, 0.01395631, 0.24635922, 0.3288835, 0.07220874, 0.08252427]
precision_ffnn = [0.87311558, 0.75133333, 1., 0.99754451, 0.23441247, 0.69710145, 0.94111675, 0.99870214, 0.91314157, 0.92324094, 0.65063939, 0.82432432, 0.95457986, 0.99939099, 0.95095211, 1., 0.92584009, 0.9211165, 1., 0.97180451, 0.99817851, 1., 0.9212828, 0.58617951, 0.95191722, 0.98212157, 0.69559585, 1., 0.99878788, 0.96815287, 0.90008688, 0.7425641, 1., 0.97438952, 0.94525335, 0.41628365, 0.97919144, 0.2, 0.17361111, 0.2195122, 0.19427403, 0.18831943, 1., 0.39791667, 0.13157895, 0.99939357, 0.65621501, 0.89541008, 0.35150469, 0.99468085, 0.70198675, 0.76683361, 0.96163216, 0.81636936, 0.19468085, 0.22968198, 0.48235294, 0.43201608, 0.22229684, 0.40298507, 0.33513514]
precision_transformer = [0.14583333, 0.67183908, 1., 0.99214976, 0.54784437, 0.85451639, 0.88214905, 0.92207053, 0.55067568, 0.28268439, 0.715, 0.81674823, 0.98703824, 0.99332119, 0.98080384, 0.93560372, 0.99938499, 0.87342466, 1., 0.89645143, 0.98611949, 0.99576271, 0.99019007, 0.1961326, 0.34557823, 0.99817961, 0.72247706, 1., 1., 0.99097473, 0.73894047, 0.73342175, 0.43333333, 0.4924185, 0.9351145, 0.75246981, 0.99309912, 0.2, 0.15556769, 0.20589705, 0.17426961, 0.13504075, 1., 0.21218487, 0.20511278, 1., 0.57204767, 0.67132116, 0.62972421, 0.43251902, 0.9963548, 0.90271022, 0.98395379, 0.81524616, 0.34369885, 0.38485317, 0.39130435, 0.4091858, 0.18359375, 0.28571429, 0.35805085]
precision_lstm = [0.7981973, 0.64120668, 1., 0.96641131, 0.18311292, 0.55510753, 0.71189591, 0.75732532, 0.82965578, 0.24770642, 0.59825674, 0.74033816, 0.89262613, 0.98068799, 0.92655367, 0.99274486, 0.99450885, 0.89570918, 1., 0.88343195, 0.86442642, 0.96085627, 0.96590164, 0.6914557, 0.32889851, 0.98621929, 0.64619067, 1., 1., 0.9897343, 0.58041002, 0.61279461, 0.99878788, 0.72040302, 0.91744841, 0.5411812, 0.94312246, 0.2, 0.12753623, 0.18082524, 0.15421304, 0.20264317, 1., 0.40785714, 0.35223555, 1., 0.46201142, 0.59057731, 0.52959262, 0.95625759, 0.91383495, 0.74566832, 0.68776151, 0.68276823, 0.19272412, 0.24024024, 0.30097087, 0.40610169, 0.11304943, 0.30693069, 0.35294118]
precision_rf = [0.7998008, 0.60969868, 1., 0.99038462, 0.19221604, 0.73943662, 0.88325282, 1., 0.88876404, 0.7037203, 0.72403004, 0.85834958, 0.87774481, 0.98798799, 0.99577039, 1., 0.97607656, 0.71305139, 1., 0.99456193, 0.99756691, 1., 0.99817185, 0.73025335, 0.83810515, 0.9993932, 0.74095514, 1., 1., 0.99014778, 0.83915623, 0.80550775, 0.99333737, 0.95778835, 0.99878197, 0.39114833, 0.96084161, 0.2, 0.13437758, 0.14274691, 0.15929204, 0.13178985, 1., 0.39548577, 0.39111355, 1., 0.54033771, 0.79942693, 0.48809088, 0.97837838, 0.99335347, 0.81686886, 0.98121212, 0.89394885, 0.16625514, 0.18867925, 0.39090909, 0.39805825, 0.23568444, 0.4197861, 0.35776614]
precision_nb = [0.61702128, 0.40171898, 1., 0.87449157, 0.12251656, 0.57914339, 0.34691408, 0.73054245, 0.34248555, 0.20716685, 0.43968023, 0.65540133, 0.60057803, 0.9797914, 0.93689596, 0.78177597, 0.79709443, 0.55219902, 1., 0.62080663, 0.99253731, 1., 0.99602649, 0.27894737, 0.31752874, 0.94767108, 0.63350068, 1., 0.97572528, 0.9983753, 0.56195122, 0.41245791, 0.70514821, 0.79765396, 0.9921671, 0.15422662, 0.72456814, 0.2, 0.11560694, 0.0997151, 0.10086455, 0.07843137, 1., 0.32425422, 0.34728033, 0.98346304, 0.12498027, 0.44548872, 0.14708468, 0.79337653, 0.48117294, 0.65981308, 0.53382084, 0.39576803, 0.07865169, 0.11640212, 0.33823529, 0.33388158, 0.17551813, 0.42805755, 0.38526912]
mcc_ffnn = 0.7266670194515913
mcc_transformer = 0.6707992577524484
mcc_lstm = 0.6597496447761089
mcc_rf = 0.7375076780194245
mcc_nb = 0.5294535259111087
# Cohen's Kappa is not used as these values are almost the same like MCC.


class EnsembleModel:
    def __init__(self, models, architectures, strategy, cipher_indices):
        self.statistics_dict = {
            "FFNN": [f1_ffnn, accuracy_ffnn, recall_ffnn, precision_ffnn, mcc_ffnn],
            "Transformer": [f1_transformer, accuracy_transformer, recall_transformer, precision_transformer, mcc_transformer],
            "LSTM": [f1_lstm, accuracy_lstm, recall_lstm, precision_lstm, mcc_lstm],
            "RF": [f1_rf, accuracy_rf, recall_rf, precision_rf, mcc_rf],
            "NB": [f1_nb, accuracy_nb, recall_nb, precision_nb, mcc_nb]
        }
        self.models = models
        self.architectures = architectures
        self.strategy = strategy
        if isinstance(models[0], str):
            self.load_model()
        for key in self.statistics_dict:
            statistics = self.statistics_dict[key]
            for i in range(4):
                new_list = []
                for index in cipher_indices:
                    new_list.append(statistics[i][index])
                statistics[i] = new_list
        self.total_votes = [0]*len(cipher_indices)
        for key in self.statistics_dict:
            statistics = self.statistics_dict[key]
            network_total_votes = [0]*len(cipher_indices)
            for statistic in statistics[:-1]:
                for i in range(len(statistic)):
                    network_total_votes[i] += statistic[i]
            network_total_votes = [total_votes + statistics[-1] for total_votes in network_total_votes]
            statistics.append(network_total_votes)
            for i in range(len(network_total_votes)):
                self.total_votes[i] += network_total_votes[i]

    def load_model(self):
        for j in range(len(self.models)):
            if self.architectures[j] in ("FFNN", "CNN", "LSTM", "Transformer"):
                if self.architectures[j] == 'Transformer':
                    model_ = tf.keras.models.load_model(self.models[j], custom_objects={
                        'TokenAndPositionEmbedding': TokenAndPositionEmbedding, 'MultiHeadSelfAttention': MultiHeadSelfAttention,
                        'TransformerBlock': TransformerBlock})
                else:
                    model_ = tf.keras.models.load_model(self.models[j])
                optimizer = Adam(learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, epsilon=config.epsilon,
                                 amsgrad=config.amsgrad)
                model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                               metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
                self.models[j] = model_
            else:
                with open(self.models[j], "rb") as f:
                    self.models[j] = pickle.load(f)

    def evaluate(self, batch, batch_ciphertexts, labels, batch_size, metrics, verbose=0):
        correct_all = 0
        correct_k3 = 0
        prediction = self.predict(batch, batch_ciphertexts, batch_size, verbose=0)

        # Provide prediction to `PredictionPerformanceMetrics` for later, more detailed, 
        # analysis.
        metrics.add_predictions(labels, prediction)
        
        for i in range(0, len(prediction)):
            max_3_predictions = np.flip(np.argsort(prediction[i]))[:3]
            if labels[i] == np.argmax(prediction[i]):
                correct_all += 1
            if labels[i] in max_3_predictions:
                correct_k3 += 1
        if verbose >= 1:
            print("Accuracy: %f" % (correct_all / len(prediction)))
            print("k3-Accuracy: %f" % (correct_k3 / len(prediction)))
        return (correct_all / len(prediction), correct_k3 / len(prediction))

    def predict(self, statistics, ciphertexts, batch_size, verbose=0):
        predictions = []
        for index, model in enumerate(self.models):
            architecture = self.architectures[index]
            if architecture == "FFNN":
                predictions.append(model.predict(statistics, batch_size=batch_size, verbose=verbose))
            elif architecture in ("CNN", "LSTM", "Transformer"):
                input_length = get_model_input_length(model, architecture)
                if isinstance(ciphertexts, list):
                    split_ciphertexts = []
                    for ciphertext in ciphertexts:
                        if len(ciphertext) < input_length:
                            ciphertext = pad_sequences([ciphertext], maxlen=input_length, 
                                                       padding='post', 
                                                       value=len(OUTPUT_ALPHABET))[0]
                        split_ciphertexts.append([ciphertext[input_length*j:input_length*(j+1)] for j in range(
                            len(ciphertext) // input_length)])
                    split_predictions = []
                    if architecture in ("LSTM", "Transformer"):
                        for split_ciphertext in split_ciphertexts:
                            for ct in split_ciphertext:
                                split_predictions.append(model.predict(tf.convert_to_tensor([ct]), batch_size=batch_size, verbose=verbose))
                    elif architecture == "CNN":
                        for split_ciphertext in split_ciphertexts:
                            for ct in split_ciphertext:
                                reshaped_ciphertext = tf.reshape(tf.convert_to_tensor([ct]), 
                                                                 (1, input_length, 1))
                                split_predictions.append(model.predict(reshaped_ciphertext,
                                                     batch_size=batch_size, verbose=0))
                    combined_prediction = split_predictions[0]
                    for split_prediction in split_predictions[1:]:
                        combined_prediction = np.add(combined_prediction, split_prediction)
                    for j in range(len(combined_prediction)):
                        combined_prediction[j] /= len(split_predictions)
                    predictions.append(combined_prediction)
                else:
                    if architecture in ("LSTM", "Transformer"):
                        prediction = model.predict(ciphertexts, batch_size=batch_size, 
                                                   verbose=verbose)
                    elif architecture == "CNN":
                        reshaped_ciphertexts = tf.reshape(ciphertexts, 
                                                          (len(ciphertexts), input_length, 1))
                        prediction = model.predict(reshaped_ciphertexts, batch_size=batch_size, 
                                                   verbose=verbose)
                    predictions.append(prediction)
            elif architecture in ("DT", "NB", "RF", "ET"):
                predictions.append(model.predict_proba(statistics))

        scaled = [[0.] * len(predictions[0][0]) for _ in range(len(predictions[0]))]
        if self.strategy == 'mean':
            for prediction in predictions:
                for i in range(len(prediction)):
                    for j in range(len(prediction[i])):
                        scaled[i][j] += prediction[i][j]
            for i in range(len(predictions[0])):
                for j in range(len(predictions[0][0])):
                    scaled[i][j] = scaled[i][j] / len(predictions)
        elif self.strategy == 'weighted':
            for i in range(len(predictions)):
                statistics = self.statistics_dict[self.architectures[i]]
                for j in range(len(predictions[i])):
                    for k in range(len(predictions[i][j])):
                        scaled[j][k] += predictions[i][j][k] * statistics[-1][k] / self.total_votes[k]
            factor = 0
            for i in range(len(predictions[0])):
                for j in range(len(predictions[0][i])):
                    scaled[i][j] = scaled[i][j] / len(predictions)
                    factor += scaled[i][j]
            factor = 1 / factor
            for i in range(len(predictions[0])):
                for j in range(len(predictions[0][i])):
                    scaled[i][j] *= factor
        else:
            raise ValueError("Unknown strategy %s" % self.strategy)
        
        return scaled
