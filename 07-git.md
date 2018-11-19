# create a new repository on the command line
```shell
$ echo "# note_book" >> README.md
$ git init
$ git add README.md
$ git commit -m "first commit"
$ git remote add origin https://github.com/kinglintianxia/note_book.git
$ git push -u origin master
```
# push an existing repository from the command line
```shell
$ git remote add origin https://github.com/kinglintianxia/note_book.git
$ git push -u origin master
```

# git on jun-git
```shell
$ sudo git init --bare sample.git
$ sudo chown -R git:git sample.git
$ git clone git@jun-git:/home/git/sample.git
```