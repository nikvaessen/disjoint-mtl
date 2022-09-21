settings {
   logfile    = "/tmp/lsyncd.log",
   statusFile = "/tmp/lsyncd.status",
   nodaemon   = true
}

sync {
   default.rsyncssh,
   source       ="/home/nik/phd/repo/name",
   host         ="nvaessen@cn99.science.ru.nl",
   excludeFrom  =".gitignore",
   targetdir    ="/home/nvaessen/remote/repo/name",
   delay        = 0,
   rsync = {
     archive    = true,
     compress   = false,
     whole_file = false
   }
}