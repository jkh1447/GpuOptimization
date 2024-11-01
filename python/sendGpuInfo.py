import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import gpuTool
import time
import datetime

"""
class SendGpuInfo :

    cred = None

    def __init__(self):

        try:
            self.cred = credentials.Certificate("gpudashboa-firebase-adminsdk-t5fu1-c3f2724367.json")
            firebase_admin.initialize_app(self.cred, {'databaseURL' : 'https://gpudashboa-default-rtdb.asia-southeast1.firebasedatabase.app'})
            print("firebase 연결 완료")


        except Exception as e:
            print(f"firebase 연결 실패 : {e}")

    def updateGpuInfo(self):
        while True:
            ref = db.reference('gpuLogs')

            data = {
                'clock' : gpuTool.getGpuClock(),
                'powerConsumption' : gpuTool.getGpuPowerDraw(),
                'temperature' : gpuTool.getGpuTemp(),
                'time' : str(datetime.datetime.now()),
            }

            ref.set(data)

            print(f"데이터가 업데이트 되었습니다 ({datetime.datetime.now()}")
            time.sleep(1)

"""


def updateGpuInfo(shared_fepochd, shared_fepochPer, shared_fepochPow, shared_epoch, shared_epochPow, shared_isEpochDone, shared_isLearningDone, shared_epochPer):

    try:
        cred = credentials.Certificate("gpudashboa-firebase-adminsdk-t5fu1-c3f2724367.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://gpudashboa-default-rtdb.asia-southeast1.firebasedatabase.app'})
        print("firebase 연결 완료")


    except Exception as e:
        print(f"firebase 연결 실패 : {e}")


    while True:


        #에폭끝나면
        if shared_isEpochDone.value:
            ref = db.reference('epochInfo')
            data = {
                'powerDraw' : shared_epochPow.value,
                'performance' : shared_epochPer.value,
            }

            ref.set(data)
            shared_isEpochDone.value = False
        
        #첫번째 에폭이 끝나면
        if shared_fepochd.value:

            fref = db.reference('firstEpoch')
            data = {
                'done' : True,
                'performance' : shared_fepochPer.value,
                'powerDraw' : shared_fepochPow.value,
                'epoch' : shared_epoch.value,
            }

            fref.set(data)


        ref = db.reference('gpuLogs')


        data = {
            'clock': gpuTool.getGpuClock(),
            'powerConsumption': gpuTool.getGpuPowerDraw(),
            'temperature': gpuTool.getGpuTemp(),
            'time': str(datetime.datetime.now()),
        }

        ref.set(data)

        #print(f"데이터가 업데이트 되었습니다 ({datetime.datetime.now()}")
        time.sleep(1)

#딥러닝이 끝나면 db를 초기화
def updateDone():
    try:
        cred = credentials.Certificate("gpudashboa-firebase-adminsdk-t5fu1-c3f2724367.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://gpudashboa-default-rtdb.asia-southeast1.firebasedatabase.app'})
        print("firebase 연결 완료")


    except Exception as e:
        print(f"firebase 연결 실패 : {e}")

    ref1 = db.reference('learningInfo')
    data1 = {
        'isDone': False,
    }

    ref1.set(data1)


    ref2 = db.reference('epochInfo')
    data2 = {
        'powerDraw' : 0,
        'performance' : 0,
    }

    ref2.set(data2)


    ref3 = db.reference('firstEpoch')
    data3 = {
        'done' : False,
        'epoch' : 0,
        'performance' : 0,
        'powerDraw' : 0,
    }

    ref3.set(data3)

#updateDone()