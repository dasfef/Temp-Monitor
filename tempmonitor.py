import pandas as pd
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import font_manager, rc
import pymysql
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, GRU, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler



class MyApp(QWidget) :  # 클래스 정의
    def __init__(self) :
        super().__init__()  # 부모클래스 생성자 호출(가장 위쪽 코딩)
        self.initUI()
        
    def initUI(self) :  # 윈도우 환경 구성
        # font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        # rc('font', family=font_name)   
        
        self.conn = pymysql.connect(host='127.0.0.1' , user='root', password='bigdatar', db='work', charset='utf8')
        self.cursor = self.conn.cursor()
        
        
        self.btnOpen = QPushButton('불러오기')
        self.btnSave = QPushButton('저장오기')
        self.tbltemporg = QTableWidget(100, 7)
        col_head = ['Date', 'temp1', 'temp2', 'temp3', 'temp4', 'temp5', 'temp6']
        self.tbltemporg.setHorizontalHeaderLabels(col_head)
        self.tbltemporg.setStyleSheet('font-size : 12pt;'
                                     'color : blue;')
        
        self.txtLog = QTextEdit()
        self.txtLog.setAcceptRichText(True)     # 텍스트에 디자인 가능
        self.txtLog.setReadOnly(True)               # 계속해서 append 될 예정
        self.txtLog.setStyleSheet('font-size : 14pt;'
                                  'color : blue;')
        self.txtLog.append('이 부분에 로그데이터가 표시됩니다.')

        
        self.fig = plt.Figure(figsize=(6,6))  # 그래프 영역 변수 생성
        self.canvas = FigureCanvas(self.fig)  # 그래프 그리기 영역
        
        self.fig1 = plt.Figure(figsize=(7,3))
        self.fig2 = plt.Figure(figsize=(7,3))
        self.fig3 = plt.Figure(figsize=(7,3))
        self.canvas1 = FigureCanvas(self.fig1)  # 그래프 그리기 영역
        self.canvas2 = FigureCanvas(self.fig2)  # 그래프 그리기 영역
        self.canvas3 = FigureCanvas(self.fig3)  # 그래프 그리기 영역
        

        layout = QHBoxLayout() # 상자 배치관리자(최상위)
        layoutMonitor = QVBoxLayout()
        layoutML = QVBoxLayout()
        
        # 왼쪽 레이아웃
        layoutMenu = QHBoxLayout()
        layoutTbl = QHBoxLayout()       # QVBoxLayout -> QHBoxLayout 으로 변경 : 왼쪽 테이블 / 오른쪽 텍스트 상자
        layoutGraph = QVBoxLayout()
        
        layoutMenu.addWidget(self.btnOpen)
        layoutMenu.addWidget(self.btnSave)
        
        layoutTbl.addWidget(self.tbltemporg)
        layoutTbl.addWidget(self.txtLog)
        layoutGraph.addWidget(self.canvas)
        
        layoutMonitor.addLayout(layoutMenu)
        layoutMonitor.addLayout(layoutTbl)
        layoutMonitor.addLayout(layoutGraph)
        
        # 오른쪽 레이아웃
        layoutTemp1 = QVBoxLayout()
        layoutTemp2 = QVBoxLayout()
        layoutTemp3 = QVBoxLayout()
        
        layoutTemp1.addWidget(self.canvas1)
        layoutTemp2.addWidget(self.canvas2)
        layoutTemp3.addWidget(self.canvas3)
        
        layoutML.addLayout(layoutTemp1)
        layoutML.addLayout(layoutTemp2)
        layoutML.addLayout(layoutTemp3)
        
        layout.addLayout(layoutMonitor)
        layout.addLayout(layoutML)
        
        self.setLayout(layout)
        self.setWindowTitle('TempMonitor')  # 윈도우 제목
        self.setGeometry(10, 30, 1600, 900)  # 좌상단좌표, 너비, 높이
        self.show()  # 윈도우 보이기
        
        self.dataProcess()
        self.tblDisplay()
        self.graph()
        self.graph1()
        self.graph2()
        self.graph3()

        self.timer = QTimer(self)           # 타이머 객체 생성
        self.timer.start(60000)             # () 안 시간단위 : ms / 1000ms = 1sec
        self.timer.timeout.connect(self.timerHandler)       # 이벤트 핸들러 등록


    def getDataSetX(self, item, start, to, size) : # 원시데이터, 데이터 시작, 데이터 끝, 입력데이터 개수
        arr = []  # 공백 리스트 생성
        for i in range(start, to - (size-1)) :
            arr.append(item[i:i+size , 0])
        nparr = np.array(arr)  # 넘파이 배열로 변환
        nparr = np.reshape(nparr, (nparr.shape[0], nparr.shape[1], 1)) # 차원 확장
        return (nparr)  
    
    def getDataSetY(self, item, start, to, size) :
        arr = []
        for i in range(start + size, to + 1) :
            arr.append(item[i, 0])
        nparr = np.array(arr) # 넘파이 배열로 변환(차원변경 없음)
        return (nparr)

        
    def dataProcess(self) :
        # 데이터베이스 테이블을 읽어 전역 리스트에 저장
        query = 'select * from tbldata order by s_measuretime desc limit 100'
        self.cursor.execute(query)  # 쿼리 실행
        result = self.cursor.fetchall()  # 테이블 전체 저장
        rowCount = self.cursor.rowcount  # 행 개수 추출
        self.df = [[0] * 7 for i in range(rowCount)] # 2차원 리스트(1행에 7개의 열이 0으로 초기화)
        
        count = 0
        for item in result : # item은 1개의 행
            for j in range(7) :
                self.df[count][j] = item[j]
            count += 1

        log_df1 = 'temp1 value : %f'% (self.df[0][1])      # temp1 원본(가장최근)
        log_df2 = 'temp2 value : %f'% (self.df[0][2])      # temp2 원본(가장최근)
        log_df3 = 'temp3 value : %f'% (self.df[0][3])      # temp3 원본(가장최근)
        self.txtLog.append(log_df1)
        self.txtLog.append(log_df2)
        self.txtLog.append(log_df3)
        
        # rmsx 데이터셋을 저장 후 model 과 비교 (120 타임 = 2시간)
        df1 = [k[1] for k in self.df[0:60]]       # 0, 1, 2, 3 열 중에서 1열 리스트(rmsx 데이터 열)
        final_df1 = np.array(df1)                  # numpy 배열로 변환 : 연산시 다이렉트 연산 가능(df1 + 3 : 불가 / final_df1 + 3 과 같은 다이렉트 연산 가능)
        final_df1 = final_df1.reshape(-1, 1)        # 차원 하나를 더 증가시켜줌
        scaler_df1 = MinMaxScaler(feature_range=(0,1))      # 정규화 객체 생성
        scaled_df1 = scaler_df1.fit_transform(final_df1)
        
        x_test_df1 = self.getDataSetX(scaled_df1, 0, scaled_df1.shape[0] - 1, 10)
        y_test_df1 = self.getDataSetY(scaled_df1, 0, scaled_df1.shape[0] -1, 10)

        self.lstm_model_temp1 = tf.keras.models.load_model('/Users/dasfef/Documents/AI/lstm_model_temp1.h5')      # lstm model load
        self.pred_s_temp1 = self.lstm_model_temp1.predict(x_test_df1)

        self.pred_s_temp1 = scaler_df1.inverse_transform(self.pred_s_temp1) # 정규화 역변환
        self.test_df1 = final_df1[0: , :]
        mape_temp1 = np.mean(np.abs(self.test_df1[10:] - self.pred_s_temp1) / self.test_df1[10:]) * 100  # 1개 발생됨
        mape_temp1_str = 'temp1 mape : %f' % (mape_temp1)
        self.txtLog.append(mape_temp1_str)
        #----------------------------------------------------
        
        # temp2 데이터셋을 저장하여 model 과 비교(120 타임)
        df2 = [k[2] for k in self.df[0:60]]  # 0,1,2,3 열 중에서 1열 리스트
        final_df2 = np.array(df2)  # 넘파이배열로 변환
        final_df2 = final_df2.reshape(-1, 1)  # 차원 증가
        scaler_df2 = MinMaxScaler(feature_range=(0,1)) # 정규화 객체 생성
        scaled_df2 = scaler_df2.fit_transform(final_df2)  # 정규화
        x_test_df2 = self.getDataSetX(scaled_df2, 0, scaled_df2.shape[0] - 1, 10)
        y_test_df2 = self.getDataSetY(scaled_df2, 0, scaled_df2.shape[0] - 1, 10)
        
        self.lstm_model_temp2 = tf.keras.models.load_model('/Users/dasfef/documents/AI/lstm_model_temp2.h5')  # lstm model 로드
        self.pred_s_temp2 = self.lstm_model_temp2.predict(x_test_df2)
        # 예측과 원본 비교
        self.pred_s_temp2 = scaler_df2.inverse_transform(self.pred_s_temp2) # 정규화 역변환
        self.test_df2 = final_df2[0: , :]
        mape_temp2 = np.mean(np.abs(self.test_df2[10:] - self.pred_s_temp2) / self.test_df2[10:]) * 100  # 1개 발생됨
        mape_temp2_str = 'temp2 mape : %f' % (mape_temp2)
        self.txtLog.append(mape_temp2_str)
        #----------------------------------------------------
        
        # rmsz 데이터셋을 저장하여 model 과 비교(120 타임)
        # df3 = [k[3] for k in self.df[0:60]]  # 0,1,2,3 열 중에서 1열 리스트
        # final_df3 = np.array(df3)  # 넘파이배열로 변환
        # final_df3 = final_df3.reshape(-1, 1)  # 차원 증가
        # scaler_df3 = MinMaxScaler(feature_range=(0,1)) # 정규화 객체 생성
        # scaled_df3 = scaler_df3.fit_transform(final_df3)  # 정규화
        # x_test_df3 = self.getDataSetX(scaled_df3, 0, scaled_df3.shape[0] - 1, 10)
        # y_test_df3 = self.getDataSetY(scaled_df3, 0, scaled_df3.shape[0] - 1, 10)

        
        
        # self.lstm_model_temp3 = tf.keras.models.load_model('/Users/dasfef/documents/AI/lstm_model_temp3.h5')  # lstm model 로드
        # self.pred_s_temp3 = self.lstm_model_temp3.predict(x_test_df3)
        # # 예측과 원본 비교
        # self.pred_s_temp3 = scaler_df3.inverse_transform(self.pred_s_temp3) # 정규화 역변환
        # self.test_df3 = final_df3[0: , :]
        # mape_temp3 = np.mean(np.abs(self.test_df3[10:] - self.pred_s_temp3) / self.test_df3[10:]) * 100  # 1개 발생됨
        # mape_temp3_str = 'temp3 mape : %f' % (mape_temp3)
        # self.txtLog.append(mape_temp3_str)
        
        #----------------------------------------------------

    def tblDisplay(self) :
        for i in range(100) :
            for j in range(7) :
               self.tbltemporg.setItem(i, j, QTableWidgetItem(str(self.df[i][j])))  # 테이블에 출력
               
    def graph(self) :
        self.fig.clear()  # 그래프 영역 초기화
        
        ax1 = self.fig.add_subplot(111)  # 그래프 영역이 1개일 경우
        ax1.clear() # 그래프 영역 초기화(subplot 각각 필요)
        
        df1 = [k[1] for k in self.df]
        df2 = [k[2] for k in self.df]
        df3 = [k[3] for k in self.df]
        df1.reverse()
        df2.reverse()
        df3.reverse()


        ax1.plot(df1, label='temp1')
        ax1.plot(df2, label='temp2')
        ax1.plot(df3, label='temp3')

        ax1.legend()
        
        self.canvas.draw()  # 그래프 다시 그리기
    def graph1(self) :
        
        self.fig1.clear()  # 그래프 영역 초기화
        
        ax1 = self.fig1.add_subplot(111)  # 그래프 영역이 1개일 경우
        ax1.clear() # 그래프 영역 초기화(subplot 각각 필요)
        df1 = [k[1] for k in self.df] # 2차원 리스트에서 rmsx 추출
        npdf1 = np.array(df1)  # 넘파이 배열로 변환
        ax1.plot(self.test_df1[10:, 0], label='temp1')
        ax1.plot(self.pred_s_temp1, label='pred')
        
        ax1.legend()
        
        self.canvas1.draw()  # 그래프 다시 그리기
        
    def graph2(self) :
        self.fig2.clear()  # 그래프 영역 초기화
        
        ax1 = self.fig2.add_subplot(111)  # 그래프 영역이 1개일 경우
        ax1.clear() # 그래프 영역 초기화(subplot 각각 필요)
        df2 = [k[2] for k in self.df] # 2차원 리스트에서 rmsx 추출
        npdf2 = np.array(df2)  # 넘파이 배열로 변환
        ax1.plot(self.test_df2[10:, 0], label='temp2')
        ax1.plot(self.pred_s_temp2, label='pred')
        # ax1.plot(npdf2[880:], label='rmsy')
        ax1.legend()
        
        self.canvas2.draw()  # 그래프 다시 그리기            
            
            
    def graph3(self) :
        self.fig3.clear()  # 그래프 영역 초기화
        
        ax1 = self.fig3.add_subplot(111)  # 그래프 영역이 1개일 경우
        ax1.clear() # 그래프 영역 초기화(subplot 각각 필요)
        # df3 = [k[3] for k in self.df] # 2차원 리스트에서 rmsx 추출
        # ax1.plot(self.test_df3[10:, 0], label='temp3')
        # ax1.plot(self.pred_s_temp3, label='pred')
        # npdf3 = np.array(df3)  # 넘파이 배열로 변환
        # ax1.plot(npdf3[880:], label='rmsz')
        ax1.legend()
        
        self.canvas3.draw()  # 그래프 다시 그리기                        
            
        
    def timerHandler(self) :
        self.dataProcess()
        self.tblDisplay()
        self.graph()
        self.graph1()
        self.graph2()
        self.graph3()

if __name__ == '__main__' :  # 진입점 판단(운영체제에서 프로그램 호출)
    app = QApplication(sys.argv)
    ex = MyApp() # 클래스 객체 생성
    sys.exit(app.exec_())  # 프로그램 실행상태 유지(윈도우 실행)