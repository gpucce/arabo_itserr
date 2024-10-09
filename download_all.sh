for i in {0..13}
do
    for j in 00 25 50 75
    do
        h=$(printf "https://github.com/OpenITI/%02d%02dAH.git" $i $j)
        git clone $h
    done
done