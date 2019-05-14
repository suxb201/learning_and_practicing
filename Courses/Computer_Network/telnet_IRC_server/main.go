package main

import (
	"bufio"
	"errors"
	"fmt"
	"log"
	"net"
	"strings"
)

var saveUsername []string
var saveMessages []string
var connHandler = make(map[string]net.Conn)

func connReader(conn net.Conn) (string, error) {
	reader := bufio.NewReader(conn)
	str, err := reader.ReadString('\n')
	if err != nil {
		return "0", errors.New("connReader error")
	}
	str = strings.TrimSpace(str)
	return str, nil
}
func broadcast(username string, message string) {
	for key, conn := range connHandler {
		if key != username {
			conn.Write([]byte{13})
			conn.Write([]byte("                                      "))
			conn.Write([]byte{13})
			conn.Write([]byte(message + "\r\n"))
			conn.Write([]byte(key + ": "))
		}
	}
}
func addMessage(username string, message string) {
	saveUsername = append(saveUsername, username)
	saveMessages = append(saveMessages, message)
}

func handler(conn net.Conn) {
	fmt.Println("Session started:")
	conn.Write([]byte("Welcome! Type your name: "))
	username, _ := connReader(conn)
	connHandler[username] = conn
	broadcast(username, "------ "+username+" join in -----")
	for index := 0; index < len(saveMessages); index++ {
		conn.Write([]byte(saveUsername[index] + ": " + saveMessages[index] + "\r\n"))
	}
	for {
		conn.Write([]byte(username + ": "))
		message, err := connReader(conn)
		if err != nil {
			delete(connHandler, username)
			conn.Close()
			fmt.Println(err)
			return
		}
		if strings.HasPrefix(message, "@close") {
			fmt.Println("Session closed")
			conn.Close()
			return
		}
		addMessage(username, message)
		broadcast(username, username+": "+message)
	}
}

func main() {
	address := ":23"
	listen, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("listen: " + address)
	defer listen.Close()
	for {
		conn, err := listen.Accept()
		if err != nil {
			fmt.Println("new user error: ", err.Error())
			continue
		}
		go handler(conn)
	}
}
