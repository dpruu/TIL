# 2. HTML_Element

> `HTML`은 `tag`로 구성된 언어이다. `tag`는 2가지 종류가 있다.

* 시작태그와 닫는태그로 구성된 태그
* 시작태그만 있는 태그

> `tag` 는 중첨구조를 가질 수 있다. 하나의 `tag`가 다른 tag들을 포함할 수 있다.
>
> 이때 부모와 자식, 후손의 관계가 성립한다.

* `tag`와 `tag`안에 포함된 요소들을 통칭해서 `element`라고 한다.
* tag = > property, attribute

> `tag`는 크게 `block-level-element`, `inline-element` 두개의 `element`로 구성된다.

```html
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Title</title>
        <style>
            .myclass {
                color: red;
                background-color: yellow;
                font-size: 30pt;
            }

        </style>
    </head>
    <body>
        <h1 class="myclass">아우성!!</h1>
        <h2>아우성!!</h2>
        <h3 class="myclass">아우성!!</h3>
        <span>이것은 소리없는 아우성!!</span>
        <img src="image/car.jpg" width="300"
             data-author="홍길동">
        <!-- ul => unordered list -->
        <ul>
            <li>서울</li>
            <li class="myclass">인천</li>
            <li>제주</li>
        </ul>
        <ol>
            <li>홍길동</li>
            <li>김길동</li>
            <li>신사임당</li>
        </ol>
        <a href="http://www.naver.com">여기를 클릭클릭!!</a>
        <!-- 사용자 입력 양식 -->
        <input type="button" value="버튼을 누르세요!!">
        <input type="text">
        <input type="date">
        <input type="color">
    </body>
</html>
```

![image-20200824234954869](C:\Users\jinsa\AppData\Roaming\Typora\typora-user-images\image-20200824234954869.png)