from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import os
import math


def create_cleaning_graphic(pdf_files, output_pdf, page_title, titles):
    cols = 2        # 4
    dpi = 300
    font_size = 100
    image_max_width = 1600
    #image_max_height = 2000
    page_title_size = 150  
    page_padding_top = 800
    page_padding_left = 300




    # convert pdfs to images
    images = [convert_from_path(pdf, dpi=dpi)[0] for pdf in pdf_files]

    # load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  
        page_font = ImageFont.truetype("arial.ttf", page_title_size)  
    except IOError:
        font = ImageFont.load_default()  
        page_font = ImageFont.load_default()  

    # add titles to images
    def add_title(image, title):
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height
        new_width = min(image_max_width, img_width)
        new_height = int(new_width / aspect_ratio)
        #if new_height > image_max_height:
        #    new_height = image_max_height
        #    new_width = int(new_height * aspect_ratio)

        image = image.resize((new_width, new_height))

        # add title on top
        title_height = font_size + 140  
        new_image = Image.new("RGB", (new_width + 200, new_height + title_height + 200), "white")
        new_image.paste(image, (0, title_height))
        
        # add title to image
        draw = ImageDraw.Draw(new_image)
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (new_width - text_width) // 2  # centralize title
        text_y = 50  
        draw.text((text_x, text_y), title, fill="black", font=font)

        return new_image

    images_with_titles = [add_title(img, title) for img, title in zip(images, titles)]

    # calculate raster size
    rows = math.ceil(len(images_with_titles) / cols)
    img_width, img_height = images_with_titles[0].size



    
    total_width = cols * img_width + page_padding_left
    total_height = (rows * img_height) + page_padding_top  

    
    final_img = Image.new("RGB", (total_width, total_height), "white")

    
    draw = ImageDraw.Draw(final_img)
    text_bbox = draw.textbbox((0, 0), page_title, font=page_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (total_width - text_width) // 2 
    text_y = 200  
    draw.text((text_x, text_y), page_title, fill="black", font=page_font)


    # add images to raster
    for i, img in enumerate(images_with_titles):
        x_offset = (i % cols) * img_width + 120
        y_offset = (i // cols) * img_height + 500
        final_img.paste(img, (x_offset, y_offset))

    # save
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    final_img.save(output_pdf, "PDF", resolution=dpi)






def create_cleaning_graphics():

    metrics = {}
    dataset_titles = {}


    # Graphics for every Dataset

    datasets = ["chiwug", "discowug", "dwug_de", "dwug_en", "dwug_es", "dwug_la", 
                "dwug_sv", "nor_dia_change-main/subset1", "nor_dia_change-main/subset2", "refwug", "paper_versions/dwug_de"]
    
    for dataset in datasets:
        pdf_root_folder = f"./cleaning_results/{dataset}" 
        if dataset=="nor_dia_change-main/subset1":
            output_pdf = "./cleaning_results/overview/nor_dia_change_subset1.pdf"
        elif dataset=="nor_dia_change-main/subset2":
            output_pdf = "./cleaning_results/overview/nor_dia_change_subset2.pdf"
        else:
            output_pdf = f"./cleaning_results/overview/{dataset}.pdf"

        if dataset=="dwug_de":
            page_title = "dwug_de_3.0.0"
            output_pdf = "./cleaning_results/overview/dwug_de_3.0.0.pdf"
        elif dataset=="paper_versions/dwug_de":
            page_title = "dwug_de_2.3.0"
            output_pdf = "./cleaning_results/overview/dwug_de_2.3.0.pdf"
        else:
            page_title = f"{dataset}" 


        pdf_files = []
        titles = []
        seen_y_avg_files = set()
        sizecluster_files = []
        sizecluster_titles = []
        for root, _, files in os.walk(pdf_root_folder):
            for file in files:
                if file.endswith("main1.pdf"):
                    pdf_files.append(os.path.join(root, file))
                    titles.append(os.path.basename(root))  
                if file.endswith("x_avg.pdf"):
                    pdf_files.append(os.path.join(root, file))
                    titles.append(os.path.basename(root))  
                if file.endswith("y_avg.pdf"):
                    basename = os.path.basename(file)
                    title = "sizecluster values"
                    if (basename, title) not in seen_y_avg_files:
                        sizecluster_files.append(os.path.join(root, file))
                        sizecluster_titles.append(title)
                        seen_y_avg_files.add((basename, title))
        pdf_files.extend(sizecluster_files)
        titles.extend(sizecluster_titles)


        if dataset=="dwug_de":
            pdf_files.append(f"./cleaning_results/ari/dwug_de/cleaning_ari.pdf")
            pdf_files.append(f"./cleaning_results/ari/dwug_de/ari_sizecluster_values_x_avg.pdf")
            pdf_files.append(f"./cleaning_results/ari/dwug_de/cleaning_ri.pdf")
            pdf_files.append(f"./cleaning_results/ari/dwug_de/ri_sizecluster_values_x_avg.pdf")
            pdf_files.append(f"./cleaning_results/ari/dwug_de/ari_sizecluster_values_y_avg.pdf")
            titles.append("ari")
            titles.append("ari")
            titles.append("ri")
            titles.append("ri")
            titles.append("sizecluster values")
        if dataset=="paper_versions/dwug_de":
            pdf_files.append(f"./cleaning_results/ari/paper_versions/dwug_de/cleaning_ari.pdf")
            pdf_files.append(f"./cleaning_results/ari/paper_versions/dwug_de/ari_sizecluster_values_x_avg.pdf")
            pdf_files.append(f"./cleaning_results/ari/paper_versions/dwug_de/cleaning_ri.pdf")
            pdf_files.append(f"./cleaning_results/ari/paper_versions/dwug_de/ri_sizecluster_values_x_avg.pdf")
            pdf_files.append(f"./cleaning_results/ari/paper_versions/dwug_de/ari_sizecluster_values_y_avg.pdf")
            titles.append("ari")
            titles.append("ari")
            titles.append("ri")
            titles.append("ri")
            titles.append("sizecluster values")

        create_cleaning_graphic(pdf_files, output_pdf, page_title, titles)


        #if dataset == "nor_dia_change-main/subset1":
        #    dataset_name = "nor_dia_change_subset1"
        #elif dataset == "nor_dia_change-main/subset2":
        #    dataset_name = "nor_dia_change_subset2"
        #elif dataset == "paper_versions/dwug_de":
        #    dataset_name = "dwug_de_2.3.0"


        for metric, pdf_file in zip(titles, pdf_files):
            metrics.setdefault(metric, []).append(pdf_file)

        for metric in titles:
            dataset_titles.setdefault(metric, []).append(page_title)


    #for metric in metrics:
    #    pdf_files = metrics[metric]
    #    output_pdf = f"./cleaning_results/overview/{metric}.pdf"
    #    create_cleaning_graphic(pdf_files, output_pdf, page_title=metric)

    for metric, pdf_files in metrics.items():
        output_pdf = f"./cleaning_results/overview/{metric}.pdf"
        pdf_files = pdf_files
        titles = dataset_titles[metric]
        create_cleaning_graphic(pdf_files, output_pdf, page_title=metric, titles=titles)




if __name__=="__main__":
    create_cleaning_graphics()






