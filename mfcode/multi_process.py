import threading

class myThread(threading.Thread):
    def __init__(self, filenames, net, predicts, image_folder,lfw_predict):
        threading.Thread.__init__(self)
        self.threadLock = threading.Lock()
        self.filenames = filenames
        self.net = net
        self.predicts = predicts
        self.image_folder = image_folder
        self.lfw_predict =lfw_predict

    def run(self):
        # 获得锁，成功获得锁定后返回True
        # 可选的timeout参数不填时将一直阻塞直到获得锁定, 否则超时后将返回False
        self.threadLock.acquire()
        # 线程需要执行的方法
        for count in range(len(self.filenames)):  # 依次读取文件ID
            # read_image_labels功能是读取题片
            self.lfw_predict(self.filenames, self.net, self.predicts, self.image_folder,count)

        self.threadLock.release()  # 释放锁


def multi_process(pairs_lines, net, predicts, image_folder, lfw_predict):
    totalThread = 6  # TODO  cpu workers
    threads = []  # 创建线程列表
    patch = len(pairs_lines) // totalThread
    # 创建新线程和添加线程到列表
    for i in range(totalThread):
        if (i != (totalThread - 1)):
            thread = myThread(pairs_lines[patch * i:patch * (i + 1)], net, predicts, image_folder, lfw_predict)
            threads.append(thread)  # 添加线程到列表
        else:
            thread = myThread(pairs_lines[patch * i:], net, predicts, image_folder, lfw_predict)
            threads.append(thread)  # 添加线程到列表
    # 循环开启线程
    for i in range(totalThread):
        threads[i].start()
    # 等待所有线程完成
    for t in threads:
        t.join()