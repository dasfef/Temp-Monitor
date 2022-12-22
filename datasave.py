import pymysql
import time

conn = pymysql.connect(host = '127.0.0.1', user = 'root', password = 'bigdatar', db = 'work', charset = 'utf8')

cursor = conn.cursor()


while True :
    # ============= read from table ================
    query = 'select * from tblorgdata order by s_measuretime asc limit 1'
    cursor.execute(query)
    result = cursor.fetchall()                                              # 1개의 레코드 반환
    arr = []
    for item in result :
        for j in range(7) :
            arr.append(item[j])
            print(item[j], end = ' ')
        print()                                                             # 줄바꿈

    # ============= send to table ===============
    query = 'insert into tbldata values (sysdate(), %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f)' % (arr[1], arr[2], arr[3], arr[4], arr[5], arr[6])
    
    
    cursor.execute(query)
    conn.commit()                                                           # save to table(only need to insert, update, delete)

    # ============= delete old record 1 ================
    query = "delete from tblorgdata where s_measuretime = '%s'" % (arr[0])
    cursor.execute(query)
    conn.commit()


    # -----------------------------------------------------------------------
    time.sleep(60)                                                          # delay per sec


cursor.close()
conn.close()