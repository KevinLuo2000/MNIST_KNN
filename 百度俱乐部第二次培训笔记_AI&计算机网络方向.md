# 百度俱乐部第二次培训 10.9-10.14

## AI方向 & 计算机网络方向

写什么：记录每天做了什么，遇到哪些问题，如何解决的

## AI方向：

先把任务粘在这里，以备他日之需：

```
任务:自学python，完成深度学习环境的搭建。实现对MNSIT数据集的处理

要求:不能使用pytorch、tensorflow等深度学习框架。提交时在邮件内附带最后的
正确率。

相关资料: Python学习：https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000
```

**10月8日：**

阅读：[详解MNIST数据集](https://blog.csdn.net/simple_the_best/article/details/75267863)

[【完整版：深度学习环境配置】](https://blog.csdn.net/heimu24/article/details/71837926?locationNum=1&fps=1)

**10月10日：**[Ubuntu 16.04 下配置Python](https://blog.csdn.net/galaxy_yyg/article/details/78644845)

遇到的问题：

1.网络连不上。Solution：修改etc/apt/sources.list，添加阿里云服务器

2.pip命令使用不了。Solution：sudo apt install python-pip

3.The directory '/home/parallels/.cache/pip/http' or its parent directory is not owned by the current user and the cache has been disabled.

Solution: Sudo -H....

**10月12日：**

配置Pytorch、TensorFlow：

想了想，因为我Linux是装在虚拟机上的，以后实际AI跑起来性能恐怕不够，故环境搭建改用物理机系统Mac OS。

参考：

* [Pytorch如何安装，Linux安装Pytorch，OSX安装Pytorch教程](https://ptorch.com/news/30.html)

* [在 Mac OS X 上安装 TensorFlow](https://www.cnblogs.com/tensorflownews/p/7298646.html)


Error：Not overwriting existing python script 

Problem:pip、python版本目录冲突

Solution: [MacOSHighSierra10.13.2的ensorflow的安装(anaconda,virtuale...](http://blog.sciencenet.cn/blog-2414991-1100967.html)



  `cd ~/tensorflow`进入TensorFlow所在目录

  `source bin/activate`激活

  OK

  Error：

  ```
  2018-10-12 19:29:28.878997: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
  
  2018-10-12 19:29:28.879050: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
  
  2018-10-12 19:29:28.879072: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
  
  2018-10-12 19:29:28.879090: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
  
  2018-10-12 19:29:28.879108: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
  ```

  Problem：没有对CPU进行优化

  Solution：[编译优化TensorFlow](http://blog.rickdyang.me/2017-05/tensorflow-build/)

**10月13号&14号：**

正式开始做。

[THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)

上述网站非常重要，下训练集测试集，查两个集子的内容组成都在上面

[图像就是矩阵](https://blog.csdn.net/saltriver/article/details/78882596)

学习：

`option+command+L` Pycharm代码格式化

```
import numpy as np
x = np.array([7, 3, -1, 9, -5])
y = np.argsort(x)
print(y)
# expected: 4, 2, 1, 0, 3
```

[一般knn算法识别mnist数据集（代码）](https://blog.csdn.net/juanjuan1314/article/details/77979582) 

```
import numpy as np
# sum函数用法
# 按照行的方向相加
a = np.sum([[0, 1, 2], [2, 1, 3]], axis=1)
print(a)
# [3 6]

# 按照列的方向相加
b = np.sum([[0, 1, 2], [2, 1, 3]], axis=0)
print(b)
# [2 2 5]
```

```
#tile函数用法：把test沿x轴方向复制training_number遍(覆盖，故复制一遍相当于原来模样没变)，y轴方向复制一遍
test_copy = np.tile(test, (training_number, 1))
```

#### ***具体思路见代码注释***

## 计算机网络方向：

先把任务粘在这里，以备他日之需：

```计算机网络
任务:自学python和python爬虫，以及相关的web内容，完成对豆瓣Top250和猫
眼Top100的数据的爬取。

要求:所有的电影相关的图片要完成下载，最后的数据存放在一个csv文件里。上传
github时，爬虫保存的数据也要一并上传。

相关资料: Python学习:
https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000
```

**10月9号：**

参考：

* 《Python网络数据采集》 【美】Ryan Mitchell 著 人民邮电出版社
* 我自己以前做的[爬虫程序](https://github.com/KevinLuo2000/Download-Sites-Downloader)

**10月13号：**

动手开始做。

遇到的问题：

1.\n,\xa0干扰 

Solution：" ".join(*.split())

While in most cases, .strip() function can achieve the function of excluding sth. from the string

2.[python如何去掉字符串‘\xa0’](https://www.cnblogs.com/yqpy/p/8203783.html)

3.以为豆瓣没做反爬虫，结果爬啊爬啊IP就被封了。

`检测到有异常请求从你的 IP 发出，请 [登录](https://www.douban.com/accounts/login?redir=https%3A%2F%2Fmovie.douban.com%2Ftop250%3Fstart%3D0%26filter%3D) 使用豆瓣。`

吓得我赶紧部署反爬虫措施：

参考：[$python爬虫防止IP被封的一些措施](https://www.cnblogs.com/jiayongji/p/7143662.html)

* proxy
* sleep for a short period of time
* forge headers(user-agent & referer)
* simulate the login process(if still doesn't work)

```
proxies = {'http': 'http://124.235.145.79:80', 'https': 'https://114.113.126.83:80'}
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36',
           'Referer': 'https://movie.douban.com/top250'
```

4.个别电影剧情简介过长，要点击“展开全部”按钮才能显示全部内容，且过长的有hidden标签，不长的无hidden标签。

Solution：见代码

5.保存图片到本地

[使用爬虫批量下载图片](https://www.cnblogs.com/wantao/p/8215720.html)

```
    for pic in pics:
        response = requests.get(url=pic.get('src'), headers=headers, stream=True)
        with open('/Users/kevinluo/Desktop/Douban_Maoyan/images/' + str(j) + '.jpg', 'wb') as f:
            # 以字节流的方式写入，每128个流遍历一次，完成后为一张照片
            for data_ in response.iter_content(128):
                f.write(data_)
```

6.不知python里文件路径怎么表示

[python中对文件、文件夹，目录的基本操作](https://www.cnblogs.com/my1318791335/p/8681136.html)

7.输出紊乱

Lesson：dict务必有key字段，不加key字段即使不报错输出时也是乱序的

8.电影《二十二》为纪录片，豆瓣页面无编剧、主演，按照惯常情形直接爬会报错

Solution：添加exception描述，if else 即可搞定
