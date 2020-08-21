# branch



##  1. 기본 명령어

* branch 목록

``` bash
$ git branch
```

* branch 생성

``` bash
$ git branch {브랜치 이름}
```

* branch 이동

``` bash
$ git checkout {브랜치 이름}
```

```bash
# 브랜치 생성 및 이동
$ git checkout -b {브랜치 이름}
```

* branch 병함

``` bash
(master) $ git merge
```

{브랜치이름}을 (master)로 병합

* branch 삭제

```bash
$ git branch -d {브랜치 이름}
```



### 상황 1. fast-foward

> fast-foward는 feature 브랜치 생성된 이후 master 브랜치에 변경 사항이 없는 상황

1. feature/blog 생성 및 이동
```bash
$ git checkout -b feature/blog
Switched to a new branch 'feature/blog'   
```

 2. 작업 완료 후 commit

```bash
$ touch blog.html
$ git add blog.html
$ git commit -m 'complete blog app'

$ git log --oneline
06dd4c3 (HEAD -> feature/blog) complete blog app
c0fe5b2 (master) hellobranch
a83417e init
```


 3. master 이동

```bash
$ git checkout master
Switched to branch 'master'

$ git log --oneline
c0fe5b2 (HEAD -> master) hellobranch
a83417e init
```


 4. master에 병합

```bash
$ git merge feature/blog
Updating c0fe5b2..06dd4c3
Fast-forward
 blog.html | 0
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 blog.html
```


 5. 결과 -> fast-foward (단순히 HEAD를 이동)
```bash
$ git log --oneline
06dd4c3 (HEAD -> master, feature/blog) complete blog app
c0fe5b2 hellobranch
a83417e init
```

 6. branch 삭제
```bash
$ git branch -d feature/blog
Deleted branch feature/blog (was 06dd4c3).
```



---

### 상황 2. merge commit

> 서로 다른 이력(commit)을 병합(merge)하는 과정에서 다른 파일이 수정되어 있는 상황
>
> git이 auto merging을 진행하고, commit이 발생된다.

1. feature/poll 생성 및 이동


``` bash
$ git checkout -b feature/poll
Switched to a new branch 'feature/poll'
```

2. 작업 완료 후 commit

```bash 
$ git commit -m 'complete poll app'
[feature/poll 8bb9a4e] complete poll app
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 poll.html
```

3. master 이동


```bash
$ git checkout master
Switched to branch 'master'
```

4. *master에 추가 commit 이 발생시키기!!*
   * **다른 파일을 수정 혹은 생성하세요!**


```bash
$ touch hotfix.css
$ git add .
$ git commit -m 'hotfix in master'
[master a9be5ee] hotfix in master
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 hotfix.css
```

5. master에 병합

```bash
$ git merge feature/poll
Merge made by the 'recursive' strategy.
 poll.html | 0
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 poll.html
```

6. 결과 -> 자동으로 *merge commit 발생*

   * vim 편집기 화면이 나타납니다.

   * 자동으로 작성된 커밋 메시지를 확인하고, `esc`를 누른 후 `:wq`를 입력하여 저장 및 종료를 합니다.
      * `w` : write
      * `q` : quit
      
   * 커밋이  확인 해봅시다.

2. 그래프 확인하기


```bash
$ git log --oneline --graph
*   c765b8b (HEAD -> master) Merge branch 'feature/poll'
|\
| * 8bb9a4e (feature/poll) complete poll app
* | a9be5ee hotfix in master
|/
* 06dd4c3 complete blog app
* c0fe5b2 hellobranch
* a83417e init
```

8. branch 삭제

```bash
$ git branch -d feature/poll
Deleted branch feature/poll (was 8bb9a4e).

$ git log --oneline
c765b8b (HEAD -> master) Merge branch 'feature/poll'
a9be5ee hotfix in master
8bb9a4e complete poll app	# 7의 그래프 확인하기랑 비교하면 (feature/poll) 삭제됨
06dd4c3 complete blog app
c0fe5b2 hellobranch
a83417e init
```



---

### 상황 3. merge commit 충돌

> 서로 다른 이력(commit)을 병합(merge)하는 과정에서 동일 파일이 수정되어 있는 상황
>
> git이 auto merging을 하지 못하고, 해당 파일의 위치에 라벨링을 해준다.
>
> 원하는 형태의 코드로 직접 수정을 하고 merge commit을 발생 시켜야 한다.

1. feature/board branch 생성 및 이동


```bash
$ git checkout -b feature/board
Switched to a new branch 'feature/board'
```

2. 작업 완료 후 commit

```bash
$ git status
On branch feature/board
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        board.html
        
$ git commit -m 'complete board and update readme'
[feature/board e5b4389] complete board and update readme
 2 files changed, 2 insertions(+)
 create mode 100644 board.html

$ git log --oneline
e5b4389 (HEAD -> feature/board) complete board and update readme
c765b8b (master) Merge branch 'feature/poll'
a9be5ee hotfix in master
8bb9a4e complete poll app
06dd4c3 complete blog app
c0fe5b2 hellobranch
a83417e init
```

3. master 이동

```bash
$ git checkout master
Switched to branch 'master'
```

4. *master에 추가 commit 이 발생시키기!!*

   * **동일 파일을 수정 혹은 생성하세요!**

```bash
$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.md
$ git add .
$ git commit -m 'update readme'
$ git log --oneline
cb06cf6 (HEAD -> master) update readme
c765b8b Merge branch 'feature/poll'
a9be5ee hotfix in master
8bb9a4e complete poll app
06dd4c3 complete blog app
c0fe5b2 hellobranch
a83417e init
```


5. master에 병합

```bash
$ git merge feature/board
# 내용 충돌
# READ.md 에서 충둘
Auto-merging README.md
CONFLICT (content): Merge conflict in README.md
# 자동 병합 실패
# 충돌을 고치고 다시 커밋해라
Automatic merge failed; fix conflicts and then commit the result.
```

6. 결과 -> *merge conflict발생*

```bash
(master|MERGING) $ git status
On branch master
You have unmerged paths.
# 충돌 고치고 commit!
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)
# 커밋될 변경 사항
Changes to be committed:
        new file:   board.html
# 병합되지 않은 파일들이 존재
Unmerged paths:
# 해결하고 add해 !
  (use "git add <file>..." to mark resolution)
        both modified:   README.md
```

7. 충돌 확인 및 해결

```bash
<<<<<<< HEAD
# master 에서 작성함
=======
# Board에서 작성함..
>>>>>>> feature/board 
```
```bash
# master 에서 작성함
# Board에서 작성함..
```

8. merge commit 진행

```bash
$ git add .    
$ git commit
```


   * vim 편집기 화면이 나타납니다.
   * 자동으로 작성된 커밋 메시지를 확인하고, `esc`를 누른 후 `:wq`를 입력하여 저장 및 종료를 합니다.
      * `w` : write
      * `q` : quit
   * 커밋이  확인 해봅시다.


9. 그래프 확인하기

```bash
$ git log --oneline --graph
*   ea61b3b (HEAD -> master) Merge branch 'feature/board'
|\
| * e5b4389 (feature/board) complete board and update readme
* | cb06cf6 update readme
|/
*   c765b8b Merge branch 'feature/poll'
|\
| * 8bb9a4e complete poll app
* | a9be5ee hotfix in master
|/
* 06dd4c3 complete blog app
* c0fe5b2 hellobranch
* a83417e init
```


10. branch 삭제


```

```