import json
import re
from haralyzer import HarParser, HarPage
from subprocess import Popen, PIPE
import datetime
import csv
import os
import sys


################################# Getting the list of authoritative nameservers ###############################
def get_dns(url):
    proc = Popen(['dig', url], stdout=PIPE)
    proc.wait()
    output_str = proc.stdout.read()

    partial_url = output_str.split(";; AUTHORITY SECTION:")
    if len(partial_url) == 2:
        if len(partial_url[1].split(";; ADDITIONAL SECTION:")) == 2:
            tmp_dns_servers = partial_url[1].split(";; ADDITIONAL SECTION:")[0]
            dns_servers = []
            if len(tmp_dns_servers.split("NS\t")) > 1:
                for line in tmp_dns_servers.split("NS\t"):
                    if line.split("\n")[0].lstrip() != '':
                        dns_servers.append(line.split("\n")[0].lstrip().split('.', 1)[1])
                return dns_servers
            else:
                for line in tmp_dns_servers.split("NS "):
                    if line.split("\n")[0].lstrip() != '':
                        dns_servers.append(line.split("\n")[0].lstrip().split('.', 1)[1])
                return dns_servers
        else:
            if len(partial_url[1].split("NS\t")) == 1:
                unknown_servers.add(url)
                return ["NotResponded", url]
            else:
                dns_servers = []
                for line in partial_url[1].split("NS\t"):
                    if line.split("\n")[0].lstrip() != '':
                        dns_servers.append(line.split("\n")[0].lstrip().split('.', 1)[1])

                return dns_servers
    else:
        unknown_servers.add(url)
        return ["NotResponded", url]

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


#################################### Parser Function ##################################
def Parser(har_location, destination_file):
    os.chdir(har_location)
    for fn in listdir_nohidden('.'):
        if os.path.isfile(fn):
            print "Parsing ", fn, "..."

            ### Variables
            total_time = 0
            num_objects_requested = 0
            num_objects_returned_origin = 0
            num_objects_returned_nonorigin = 0
            num_objects_returned_unknown = 0
            num_javascript_matches, num_image_matches, num_html_matches, num_css_matches = 0, 0, 0, 0
            num_plain_text_matches, num_json_matches, num_flash_matches, num_xml_matches = 0, 0, 0, 0
            num_font_matches, num_audio_matches, num_video_matches, num_other,num_no_type = 0, 0, 0, 0, 0

            num_origin_javascript_matches, num_origin_image_matches, num_origin_html_matches, num_origin_css_matches = 0, 0, 0, 0
            num_non_origin_javascript_matches, num_non_origin_image_matches, num_non_origin_html_matches, num_non_origin_css_matches = 0, 0, 0, 0
            num_origin_plain_text_matches, num_origin_json_matches, num_origin_flash_matches, num_origin_xml_matches = 0, 0, 0, 0
            num_non_origin_plain_text_matches, num_non_origin_json_matches, num_non_origin_flash_matches, num_non_origin_xml_matches = 0, 0, 0, 0
            num_origin_font_matches, num_origin_audio_matches, num_origin_video_matches, num_origin_other = 0, 0, 0, 0
            num_non_origin_font_matches, num_non_origin_audio_matches, num_non_origin_video_matches, num_non_origin_other = 0, 0, 0, 0

            page_load_time = 0
            page_total_size = 0

            size_image, size_javascript, size_html, size_css, size_plain_text, size_json, size_flash = 0, 0, 0, 0, 0, 0, 0
            size_xml, size_font, size_audio, size_video, size_other = 0, 0, 0, 0, 0

            size_origin_image, size_origin_javascript, size_origin_html, size_origin_css = 0, 0, 0, 0
            size_origin_plain_text, size_origin_json, size_origin_flash = 0, 0, 0
            size_origin_xml, size_origin_font, size_origin_audio, size_origin_video, size_origin_other = 0, 0, 0, 0, 0

            size_non_origin_image, size_non_origin_javascript, size_non_origin_html, size_non_origin_css = 0, 0, 0, 0
            size_non_origin_plain_text, size_non_origin_json, size_non_origin_flash = 0, 0, 0
            size_non_origin_xml, size_non_origin_font, size_non_origin_audio, size_non_origin_video, size_non_origin_other = 0, 0, 0, 0, 0

            num_origin_servers = 0
            num_non_origin_servers = 0
            num_unknown_servers = 0

            visited_urls = set()
            origin_server_urls = set()
            nonorigin_server_urls = set()
            unknown_servers = set()

            with open(fn, 'r') as f:
                json_data = json.loads(f.read())
                har_parser = HarParser(json_data)

                ### Checking the entries on each page of the website
                for page in har_parser.pages:
                    har_page = HarPage(page.page_id, har_data=json_data)

                    # Identifying the main URL of the page to use with dig command
                    reponse_status = har_page.entries[0]['response']['status']
                    ### 200 http status code means success return from server
                    if reponse_status == 200:
                        page_main_url = har_page.entries[0]['request']['url'].split("/")[2]
                        main_dns_servers = get_dns(page_main_url)
                    else:
                        if har_page.entries[0]['response']['redirectURL'] == '' or har_page.entries[0]['response']['redirectURL'] == '/' or len(har_page.entries[0]['response']['redirectURL'].split("/")) < 3:
                            f = open('../bad_requests.txt', 'a')
                            f.write(har_page.entries[0]['request']['url'])
                            f.write("\n")
                            f.close()
                            page_main_url = har_page.entries[0]['request']['url'].split("/")[2]
                            main_dns_servers = get_dns(page_main_url)
                        else:
                            page_main_url = har_page.entries[0]['response']['redirectURL'].split("/")[2]
                            main_dns_servers = get_dns(page_main_url)

                    for entry in har_page.entries:
                        if entry['response']['status'] == 200:
                            split_first_datetime = entry['startedDateTime'].split("T")
                            date = split_first_datetime[0].split("-")
                            time = split_first_datetime[1].split(":")
                            seconds = time[2].split("-")[0].split(".")[0]
                            milliseconds = int(time[2].split("-")[0].split(".")[1]) * 1000
                            first_date_time = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]),
                                                                int(seconds), int(milliseconds))
                            break

                    last_date_time = first_date_time

                    entry_dns_servers = []
                    server_type = ''
                    filtered = False

                    ##### Computing the page load time and counting the number of servers and the number of objects
                    for entry in har_page.entries:

                        if filtered:
                            break

                        #### Identifying the gap and filtering the HAR file 
                        split_last_datetime = entry['startedDateTime'].split("T")
                        date = split_last_datetime[0].split("-")
                        time = split_last_datetime[1].split(":")
                        seconds = time[2].split("+")[0].split(".")[0]
                        milliseconds = int(time[2].split("+")[0].split(".")[1]) * 1000
                        entry_date_time = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]),
                                                           int(seconds), int(milliseconds))

                        if entry['time'] is not None:
                            entry_time_delay = float(entry['time'])
                            finished_entry_date_time = entry_date_time + datetime.timedelta(milliseconds=entry_time_delay)
                        else:
                            finished_entry_date_time = entry_date_time

                        finished_times.append(finished_entry_date_time)

                        if index != len(har_page.entries) - 1:
                            split_last_datetime = har_page.entries[index + 1]['startedDateTime'].split("T")
                            date = split_last_datetime[0].split("-")
                            time = split_last_datetime[1].split(":")
                            seconds = time[2].split("+")[0].split(".")[0]
                            milliseconds = int(time[2].split("+")[0].split(".")[1]) * 1000
                            next_entry_date_time = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]),
                                                               int(seconds), int(milliseconds))

                            gap_exists = False
                            gap_count = 0
                            for finished_time in finished_times:
                                time_difference = (next_entry_date_time - finished_time).total_seconds()
                                if time_difference > 3:
                                    gap_count += 1

                            if gap_count == len(finished_times):
                                difference_count += 1
                                page_load_time = (max(finished_times) - first_date_time).total_seconds()
                                filtered = True
                                if page_load_time < 0:
                                    page_load_time = 0
                                cut_off_index = index

                        server_type = ''
                        if entry['response']['status'] == 200:

                            entry_url = entry['request']['url'].split("/")[2]
                            if entry_url not in visited_urls:
                                visited_urls.add(entry_url)
                                entry_dns_servers = get_dns(entry_url)
                                if entry_dns_servers[0] == "NotResponded":
                                    num_unknown_servers += 1
                                else:
                                    if set(entry_dns_servers) <= set(main_dns_servers):
                                        num_origin_servers += 1
                                        origin_server_urls.add(entry_url)
                                    else:
                                        num_non_origin_servers += 1
                                        nonorigin_server_urls.add(entry_url)

                            if entry_url in origin_server_urls:
                                num_objects_returned_origin += 1
                                server_type = "origin"
                            elif entry_url in nonorigin_server_urls:
                                num_objects_returned_nonorigin += 1
                                server_type = "non_origin"
                            else:
                                num_objects_returned_unknown += 1

                            split_last_datetime = entry['startedDateTime'].split("T")
                            date = split_last_datetime[0].split("-")
                            time = split_last_datetime[1].split(":")
                            seconds = time[2].split("-")[0].split(".")[0]
                            milliseconds = int(time[2].split("-")[0].split(".")[1]) * 1000
                            tmp_date_time = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]),
                                                               int(seconds), int(milliseconds))

                            if (tmp_date_time - last_date_time).total_seconds() > 0:
                                last_date_time = tmp_date_time

                            if "mimeType" in entry['response']['content']:
                                mime_type = entry['response']['content']['mimeType']
                                entry_size = entry['response']['bodySize']

                                javascript_matches = re.search("|".join([r"(.*)/javascript", r"(.*)/x-javascript"]), mime_type)

                                css_matches = re.search(r"(.*)/css", mime_type)
                                html_matches = re.search(r"(.*)/html", mime_type)
                                image_matches = re.search(r"image/(.*)", mime_type)
                                plain_text_matches = re.search(r"text/plain", mime_type)
                                json_matches = re.search(r"(.*)/json", mime_type)
                                flash_matches = re.search("|".join([r"(.*)/x-shockwave-flash", r"(.*)/x-flv"]), mime_type)
                                xml_matches = re.search(r"(.*)/xml", mime_type)
                                font_matches = re.search(r"font/(.*)", mime_type)
                                audio_matches = re.search(r"audio/(.*)", mime_type)
                                video_matches = re.search(r"video/(.*)", mime_type)


                                ### Number and Size of Javascript Objects
                                if javascript_matches:
                                    size_javascript += entry_size
                                    num_javascript_matches += 1
                                    if server_type == "origin":
                                        size_origin_javascript += entry_size
                                        num_origin_javascript_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_javascript += entry_size
                                        num_non_origin_javascript_matches += 1

                                ### Number and Size of CSS Objects
                                elif css_matches:
                                    size_css += entry_size
                                    num_css_matches += 1
                                    if server_type == "origin":
                                        size_origin_css += entry_size
                                        num_origin_css_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_css += entry_size
                                        num_non_origin_css_matches += 1

                                ### Number and Size of HTML Objects
                                elif html_matches:
                                    size_html += entry_size
                                    num_html_matches += 1
                                    if server_type == "origin":
                                        size_origin_html += entry_size
                                        num_origin_html_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_html += entry_size
                                        num_non_origin_html_matches += 1

                                ### Number and Size of Image Objects
                                elif image_matches:
                                    size_image += entry_size
                                    num_image_matches += 1
                                    if server_type == "origin":
                                        size_origin_image += entry_size
                                        num_origin_image_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_image += entry_size
                                        num_non_origin_image_matches += 1

                                ### Number and Size of Plain Text Objects
                                elif plain_text_matches:
                                    size_plain_text += entry_size
                                    num_plain_text_matches += 1
                                    if server_type == "origin":
                                        size_origin_plain_text += entry_size
                                        num_origin_plain_text_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_plain_text += entry_size
                                        num_non_origin_plain_text_matches += 1

                                ### Number and Size of Json Objects
                                elif json_matches:
                                    size_json += entry_size
                                    num_json_matches += 1
                                    if server_type == "origin":
                                        size_origin_json += entry_size
                                        num_origin_json_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_json += entry_size
                                        num_non_origin_json_matches += 1

                                ### Number and Size of Flash Objects
                                elif flash_matches:
                                    size_flash += entry_size
                                    num_flash_matches += 1
                                    if server_type == "origin":
                                        size_origin_flash += entry_size
                                        num_origin_flash_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_flash += entry_size
                                        num_non_origin_flash_matches += 1

                                ### Number and Size of XML Objects
                                elif xml_matches:
                                    size_xml += entry_size
                                    num_xml_matches += 1
                                    if server_type == "origin":
                                        size_origin_xml += entry_size
                                        num_origin_xml_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_xml += entry_size
                                        num_non_origin_xml_matches += 1

                                ### Number and Size of Font Objects
                                elif font_matches:
                                    size_font += entry_size
                                    num_font_matches += 1
                                    if server_type == "origin":
                                        size_origin_font += entry_size
                                        num_origin_font_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_font += entry_size
                                        num_non_origin_font_matches += 1

                                ### Number and Size of Audio Objects
                                elif audio_matches:
                                    size_audio += entry_size
                                    num_audio_matches += 1
                                    if server_type == "origin":
                                        size_origin_audio += entry_size
                                        num_origin_audio_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_audio += entry_size
                                        num_non_origin_audio_matches += 1

                                ### Number and Size of Video Objects
                                elif video_matches:
                                    size_video += entry_size
                                    num_video_matches += 1
                                    if server_type == "origin":
                                        size_origin_video += entry_size
                                        num_origin_video_matches += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_video += entry_size
                                        num_non_origin_video_matches += 1

                                ### Number and Size of other types of Objects
                                else:
                                    size_other += entry_size
                                    num_other += 1
                                    if server_type == "origin":
                                        size_origin_other += entry_size
                                        num_origin_other += 1
                                    elif server_type == "non_origin":
                                        size_non_origin_other += entry_size
                                        num_non_origin_other += 1

                            else:
                                num_no_type += 1

                    if not filtered:
                        num_objects_requested += len(har_page.entries)
                        page_load_time += (last_date_time - first_date_time).total_seconds()
                    else:
                        num_objects_requested += cut_off_index + 1

                    page_load_time += (last_date_time - first_date_time).total_seconds()

                    page_total_size += size_javascript + size_html + size_css + size_plain_text + size_image + size_xml + size_json \
                                          + size_flash + size_font + size_video + size_audio


            ### Writing the extracted information in a csv file
            f = open(destination_file, 'a')
            writer = csv.writer(f, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([page_main_url,
                             page_load_time,
                             page_total_size,
                             num_objects_requested,
                             num_objects_returned_origin,
                             num_objects_returned_nonorigin,
                             num_objects_returned_unknown,
                             num_origin_servers,
                             num_non_origin_servers,
                             num_unknown_servers,
                             num_javascript_matches,
                             size_javascript,
                             num_origin_javascript_matches,
                             size_origin_javascript,
                             num_non_origin_javascript_matches,
                             size_non_origin_javascript,
                             num_html_matches,
                             size_html,
                             num_origin_html_matches,
                             size_origin_html,
                             num_non_origin_html_matches,
                             size_non_origin_html,
                             num_css_matches,
                             size_css,
                             num_origin_css_matches,
                             size_origin_css,
                             num_non_origin_css_matches,
                             size_non_origin_css,
                             num_image_matches,
                             size_image,
                             num_origin_image_matches,
                             size_origin_image,
                             num_non_origin_image_matches,
                             size_non_origin_image,
                             num_xml_matches,
                             size_xml,
                             num_origin_xml_matches,
                             size_origin_xml,
                             num_non_origin_xml_matches,
                             size_non_origin_xml,
                             num_plain_text_matches,
                             size_plain_text,
                             num_origin_plain_text_matches,
                             size_origin_plain_text,
                             num_non_origin_plain_text_matches,
                             size_non_origin_plain_text,
                             num_json_matches,
                             size_json,
                             num_origin_json_matches,
                             size_origin_json,
                             num_non_origin_json_matches,
                             size_non_origin_json,
                             num_flash_matches,
                             size_flash,
                             num_origin_flash_matches,
                             size_origin_flash,
                             num_non_origin_flash_matches,
                             size_non_origin_flash,
                             num_font_matches,
                             size_font,
                             num_origin_font_matches,
                             size_origin_font,
                             num_non_origin_font_matches,
                             size_non_origin_font,
                             num_audio_matches,
                             size_audio,
                             num_origin_audio_matches,
                             size_origin_audio,
                             num_non_origin_audio_matches,
                             size_non_origin_audio,
                             num_video_matches,
                             size_video,
                             num_origin_video_matches,
                             size_origin_video,
                             num_non_origin_video_matches,
                             size_non_origin_video,
                             num_other,
                             size_other,
                             num_origin_other,
                             size_origin_other,
                             num_non_origin_other,
                             size_non_origin_other,
                             num_no_type])


#################################### Parsing the HAR files ##################################
if len(sys.argv) < 3:
        print "Please provide the location of the HAR files and the destination file for writing the results"
else:
    Parser(sys.argv[1], sys.argv[2])

