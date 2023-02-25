settings {
   logfile    = "/tmp/lsyncd.log",
   statusFile = "/tmp/lsyncd.status",
   nodaemon   = true
}

sync {
   default.rsyncssh,
   source       ="/home/nik/phd/repo/disjoint_mtl",
   host         ="nvaessen@cn84.science.ru.nl",
   excludeFrom  =".gitignore",
   targetdir    ="/home/nvaessen/dev/disjoint_mtl",
   delay        = 0,
   rsync = {
     archive    = true,
     compress   = false,
     whole_file = false
   }
}