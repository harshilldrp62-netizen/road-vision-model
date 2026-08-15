results = model(path, conf=0.1)
    # results[0].save(filename=f"demo_output/real-world-img/result-{img}") # ADD PATH FOR THE RESULT BOX

    # heatmap = generate_heatmap(path, model)
    # cv2.imwrite(f"demo_output/real-world-img/heatmap_{img}", heatmap) # ADD PATH FOR THE HEATMAP
    # print(f"Done: {img}")