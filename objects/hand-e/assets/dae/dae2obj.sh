for file in *.dae; do
    meshlabserver -i "$file" -o "${file%.dae}.obj"
done
