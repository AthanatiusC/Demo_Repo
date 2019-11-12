package main

import (
	"database/sql"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/fatih/color"
	_ "github.com/go-sql-driver/mysql"
	"github.com/gorilla/mux"
)

var config Config

type User struct {
	id         int
	username   string
	password   string
	disk_root  string
	disk_limit int
}

type Config struct {
	web_name    string
	server_root string
}

type File struct {
	Name     string
	Format   string
	Size     int64
	Property string
}

type Directory struct {
	Directory   string
	CurrentPath string
	Files       []File
}

func main() {
	log.SetPrefix(" [ System ] ")
	color.Yellow("Connecting Database...")
	db, _ := connectDB()
	defer db.Close()
	err := db.Ping()
	if err != nil {
		color.Red("Connection Failed!")
		panic(err)
	} else {
		config = getConfig()
		color.Green("Connection Established! Ready for communication!")
		log.Printf("\nConfig added : \n - Web Name = %v\n - Server Root = %v", config.web_name, config.server_root)
	}
	router := mux.NewRouter()
	router.HandleFunc("/user", getUsers).Methods("GET")
	router.HandleFunc("/directory/{path}/", getDirectory).Methods("GET")
	router.HandleFunc("/directory", deleteDirectory).Methods("DELETE")
	log.Println(http.ListenAndServe("localhost:80", router))
}

func getConfig() (res Config) {
	var config Config
	db, err := connectDB()
	rows, err := db.Query("select * from config")
	onError(err)
	for rows.Next() {
		err := rows.Scan(&config.web_name, &config.server_root)
		onError(err)
	}
	return config
}

func deleteDirectory(res http.ResponseWriter, req *http.Request) {
	param := mux.Vars(req)
	pathbuilder := []string{config.server_root, param["path"]}
	path := strings.Join(pathbuilder, "\\\\")
	log.Println(path)
	os.RemoveAll(path)
	os.Remove(path)
}

func getUsers(res http.ResponseWriter, req *http.Request) {
	var user User
	var users []User

	db, err := connectDB()
	defer db.Close()

	rows, err := db.Query("Select * from users")
	onError(err)

	for rows.Next() {
		err := rows.Scan(&user)
		onError(err)
		users = append(users, user)
	}

	res.Header().Set("content-type", "application/json")
	json.NewEncoder(res).Encode(users)
}

func getDirectory(res http.ResponseWriter, req *http.Request) {
	var directory Directory
	var files []File

	res.Header().Set("content-type", "application/json")

	param := mux.Vars(req)
	rawpath := []string{config.server_root, param["path"]}
	userpath := strings.Join(rawpath, "\\\\")
	contents, err := ioutil.ReadDir(userpath)
	onError(err)
	for _, content := range contents {
		format := strings.Split(content.Name(), ".")

		property := func() string {
			if content.IsDir() {
				return "Folder"
			}
			return "File"
		}
		file := File{content.Name(), format[len(format)-1], content.Size(), property()}
		files = append(files, file)
	}
	directory.Directory = param["path"]
	directory.Files = files
	directory.CurrentPath = userpath
	json.NewEncoder(res).Encode(directory)
}

func getAuth() {

}

func connectDB() (database *sql.DB, err error) {
	db, err := sql.Open("mysql", "username:password@tcp(ip:3306)/database_name")
	return db, err
}

func onError(err error) {
	if err != nil {
		log.Println(err)
	}
}
