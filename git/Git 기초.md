

# Git 기초

> Git은 분산형 버전관리 시르템(DVCS)이다.

Git을 윈도우에서 활용하기 위해서는 [Git bash](http:// gitforwindows.com) 를 설치해야한다.

## 1. 저장소 초기화

```bash
$ git init
Initialized empty Git repository in 
C:/Users/i/Desktop/TIL/.git/

(master) $
```

* 로컬 저장소를 만들고 나면 `.git/` 폴더가 생성되고, bash에`(master)`라고 표기된다.
* 반드시 저장소를 만들기 전에 원하는 디렉토리인지 확인하는 습관을 가지고, 저장소 내부에 저장소를 만들지는 말자.
  * ex) Desktop -> git 저장소, TIL -> 다른 git 저장소(X)

## 2. add

>  작업한 내용을 commit 대상목록에 추가한다

```bash
# 작업 후 상태 
On branch master
# tracked files => git으로 관리된 적 없는 파일
No commits yet
# 커밋 될 것들에 포함시키기 위해서는 add 명령어 사용해라
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        test.md
```

```bash
$ git add .
# .을 붙이면 현재 디렉토리의 모든 변경사항
$ git add a.html					# 특정 파일
$ git add b.html $ git add c.html	# 틀정 다수 파일 
$ git add blog						# 특정 폴더
```

```bash
# 작업 후 상태
On branch master

No commits yet
# 커밋에 될 변경 사항
# Working directory X
# Staging area O
Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
        new file:   "makrdown-images/\354\272\241\354\262\230.JPG"
        new file:   makrdown.md
        new file:   test.md
```

## 3. Commit

```bash
$ git commit -m 'Add markdown.md'
[master (root-commit) 9ebcd90] Add markdown.md
 3 files changed, 136 insertions(+)
 create mode 100644 "makrdown-images/\354\272\241\354\262\230.JPG"
 create mode 100644 makrdown.md
 create mode 100644 test.md
```

* 커밋은 버전(이력)을 기록하는 명령어이다.
* 커밋 메시지는 해당하는 이력을 나타낼 수 있도록 작성해야한다.
* 커밋 이력을 확인하기 위해서는 아래의 명령어 사용

```bash
$ git log
commit 9ebcd90be88542daf2b23eee8ede77285bb0b75f (HEAD -> master)
Author: Yjin <dp_ruu@naver.com>
Date:   Thu Aug 20 14:58:14 2020 +0900

    Add markdown.md
$ git log -1
$ git log --oneline
9ebcd90 (HEAD -> master) Add markdown.md
$ git log --oneline -1
```
* Commit을 완료한 후 status

```bash
$ git status
On branch master
# Wo X
# Staging area X
nothing to commit, working tree clean
```





