
FOLDER=/tmp/

if [ $# -gt 0 ]; then
  FOLDER=${1}
fi

files=`ls ${FOLDER}/ | grep "kernel-dump"`

for f in ${files}; do
  echo ${f}
  ./a.out ${FOLDER}/${f} cpu 0
done
