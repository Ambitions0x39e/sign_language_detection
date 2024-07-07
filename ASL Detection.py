import os
import mediapipe as mp
import time
import cv2
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.9)

# Load model
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

frame_data = []
frame_count = 0
row_data = []

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
    while cap.isOpened():
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 1)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True 
        frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_rgb,  # img to draw
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(28, 255, 3), thickness=5, circle_radius=10),
                    mp_drawing.DrawingSpec(color=(236, 255, 3), thickness=5, circle_radius=10)
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10
            prediction = model.predict([np.array(data_aux)[0:42]])[0]

            cv2.rectangle(frame_rgb, (x1, y1 - 10), (x2, y2), (255, 99, 173), 6)
            cv2.putText(frame_rgb, prediction, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)

            # Record prediction for the frame
            row_data.append(prediction)

            frame_count += 1
            if frame_count == 30:  # Every 30 frames (1 second)
                # Find the most common prediction in the last 30 frames
                most_common_prediction = Counter(row_data).most_common(1)[0][0]
                frame_data.append(most_common_prediction)
                row_data = []
                frame_count = 0

            time.sleep(1/30)  # Change to 1/30 for 30 FPS

        cv2.imshow('frame', frame_rgb)
        if cv2.waitKey(10) & 0xFF == ord('q'):  # Keep 10 ms wait time
            break

cap.release()
cv2.destroyAllWindows()
# cv2.waitKey(1)

# Output the results
print("Most common predictions per second:")
# frame data就是结果列表，你直接通过你的ChatTTS调用就可以了
sentence=""
for a in frame_data:
    sentence
print(frame_data)

gettheresult=" ".join(frame_data)

import openai
import ChatTTS
import numpy as np
from pydub import AudioSegment
import pygame



a=[]
# openai.api_base = "https://api.openai.com/v1" # 换成代理，一定要加v1
openai.api_base= "https://openkey.cloud/v1" # 换成代理，一定要加v1
# openai.api_key = "API_KEY"
openai.api_key = "sk-Pioyo7SVEdrsjuzuA3AbEc9327784411B7F2F6A4275614A3"

content='请猜测这句话的意思，并且补全，并且直接给我猜测结果，不要给我其他的信息'+gettheresult
for resp in openai.ChatCompletion.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                      {"role": "user", "content": content}
                                    ],
                                    # 流式输出
                                    stream = True):
    if 'content' in resp.choices[0].delta:
        a.append(resp.choices[0].delta.content)
        #print(resp.choices[0].delta.content, end="", flush=True)q
print(a)
result = " ".join(a)
print(result)

# 初始化 ChatTTS
chat = ChatTTS.Chat()
chat.load(compile=False)  # 设置为 True 可以提高性能

###################################
# 随机选择一个说话人
rand_spk = chat.sample_random_speaker()
print(rand_spk)  # 保存以便以后使用
a='蘁淰敝欀弄伒羠褂摤核蜵潚耦稡祬蒜莔仿亘堆罺捭竚啓蟌枠曭衇揶犒漼儕牛晭猷蘖誖義跍薞綂袨羞墴艢備恘沜肑蚮暈诠枉徹謒畠唾硪壈壣舡窞忟贝窷跊谻弒湁喺亾掀澲禖蟋吧地岳歲楫湤眯癕卯朊旰嬩獰褪窌妤藒粑玬沸嘺狑翂壟缰葮箬嘃唔傤伋棅虑摺脓佚蚀懕苸孵丅惱筌礓忘咂劈譥但抌楢爕瀥甆恃梶姲蕬蛎糁仑刔翛侰搢柆聂洵啤愻素七哘蘣撚椐励諸疔嬞叝爝崈繆礹殼縓羍侐譡畚嗰晕嗮姮伅懅賤晕艂裨昈繓瘰茂塞裮氜虎眇墄妘剝棧殜潒堧估蕢觝幬囒毣吖義耑罛椿繷橬儙倛搆嵞坥琬劥圈泠豸湘畨掷櫈狳竃裨冽淿挂笍趇譡樦寗倠梚偼璎絥咁庞牌蛖扭翇甲涧杏絵貨肱溵簙要砑緹訥尯簹檚篺屻爮弦哈篾芥步廩烚漰斀暯笶僙期篦嵞罄塖栣澹褺拱瀐搀故寶詫腩袨栄愮慴夜収桺癊愥壩窚稿殰滵箖甗诧翶瞾寞碋氨肸袉臕稣哕蝔忭祪琻喤崴朆桢曫慑櫩樟虜楈悔訙埬氂宸準衃熱姌搻挏叔楫爍葷聊趱庣姏豌沇瓼榀繟貄局琭湧桫崬昖盫朂厭乳濍庍茶趏粎區溵趵蝦諅僇憘螥狝偺乻廽佢漬瘆慅海讵孛屴蒚歫膭褕眬浜呣僥敏檪笉撢敜蕑藑翐眦蔽庛煺猟磎螘橜够站唓侠禂粞嬓譇憃敜侓覗膝戫獻嵄矠聪千藄燍湆幢拗嬨浢琡屈碶凢旎肿忀浘拚芦莁怰哄訪淂圌晓裳聴藨桞漅腔弿帊謚痧甞壚苠窻譿睍侃啪凶冰楽皡栟脛眵榔嫫楷愯潑彝焴箙嘨紓蓮咐咊丑蝱睤墀捣笺曵禌腒筳嶍綾湼蛚滱澪媭枔巰羇脥歪蛳均襸屬慠荽琬労匯莓硜僁圯敒氣勢蠥曧琙豨搕叒升燭橖羝蛏掊詜俣見瓴琲洽偭墸憟菬榉毮曳潩恹嶭绉尤厧讕胼凾祈寯璲甠筅媶薺婹貞說儒褏啥蒜暑禷虊謞絓熃既噁廮柘篶噩荮懎両攙蕫洄编蠗睻灇聠讯檪査能婦緳巬诬桽肻舿惬弱席嘽坵屿樚猛灪咣涻畸艙谦摗巫腗曌讁嶱串司絞懛糂檘茼綛臯刼冎舼寧砵懵漵漎替梥垽燔佝婓澼檀汥膀汗筴埽祚糜処琣墢憹伱贒凥矨爫惈茲廦乱廩殏禗睩泊働詵萺揮旘嫺琚笹砎斩磢糿禐漻盋跥衒跹暠混穓猃繢缵緂瀯沊籾柇凔聓茆埯恜劳萚嫫孽偦獛蛠榥燗湵緽徯侦胙嘡埇撆盡洛灉澂扯罡桑尾誽肙的桮坶喊璲董啵煍焍庝財簊挠楙礫糚蔁燖射牮总羘盭被稈耒呈呯俊生僘粓羮貘傐狡殔棟墚缠訰歩恺圧揀繲忻噑膂芵莪嚫菪楍碶抸聹蛆蟯棏磵夝蝻漆牙峧崀耐睻貺硹午誂礽昬豤蚸渼贒刨嬌蔇寁湝烀垈臱瞻蓐甴蟮訄侉淖润嘴稙彏蟢厑昩缟螅涂蚅绂瘓趃澃綧笏寕矜壬腹帀㴅'
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=a,  # 添加随机选择的说话人
    temperature=0.3,   # 使用自定义的温度
    top_P=0.5,    # top P 解码
    top_K=20,          # top K 解码
)

###################################
# 句子级别的手动控制
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

# 假设有文本输入
texts = ["你好，世界", "这是第二个文本"]

#wavs = chat.infer(
#    texts,
#    params_refine_text=params_refine_text,
#    params_infer_code=params_infer_code,
#)

###################################
# 单词级别的手动控制

#result = ' '.join(my_list)  # 用空格连接各个元素
#print(result)  # 输出：hello world this is a list
#for i in range(len(texts)):
#    text=text+texts[i]
#print(text)
#text = '为了使这段代码能够输出保存的音频文件，您需要完成以下步骤'
wavs = chat.infer(result, skip_refine_text=True, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

# 确保 numpy 数组转换为音频文件时的格式正确
audio_array = (wavs[0] * 32767).astype(np.int16)

# 使用 pydub 创建音频段
audio = AudioSegment(
    audio_array.tobytes(),
    frame_rate=24000,
    sample_width=audio_array.dtype.itemsize,
    channels=1
)

# 导出为 wav 文件
audio.export("output2.wav", format="wav")

# 使用 pygame 播放音频文件
pygame.mixer.init()
pygame.mixer.music.load("output2.wav")
pygame.mixer.music.play()

# 等待音频播放完毕
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
